################################################################################
# shape2d/affine.py: Affine transform class and support functions
################################################################################

from __future__ import division

import numpy as np
from polymath import Scalar, Pair, Matrix3, Qube

class Affine(object):
    """An class that describes 2-D affine transformations. These are transforms
    of the form:
        x' = a x + b y + c
        y' = d x + e y + f
    """

    #===========================================================================
    def __init__(self, a, b, c, d, e, f):
        """Constructor for an Affine transform."""

        self.a = Scalar.as_scalar(a).as_float()
        self.b = Scalar.as_scalar(b).as_float()
        self.c = Scalar.as_scalar(c).as_float()
        self.d = Scalar.as_scalar(d).as_float()
        self.e = Scalar.as_scalar(e).as_float()
        self.f = Scalar.as_scalar(f).as_float()

        # Indexed by recursive = True/False
        self.scalars = (self.a, self.b, self.c, self.d, self.e, self.f)
        self.scalars_wod = (self.a.without_derivs(),
                            self.b.without_derivs(),
                            self.c.without_derivs(),
                            self.d.without_derivs(),
                            self.e.without_derivs(),
                            self.f.without_derivs())

        self.abcdef = [self.scalars_wod, self.scalars]
        self.shape = Qube.broadcasted_shape(*self.scalars)

        self.inverted = None        # Use to track the inverse transform

    #===========================================================================
    def inverse(self):
        """The Affine transform that undoes this one."""

        if self.inverted:
            return self.inverted

        # Fill in inverse transform parameters
        # x' = a x + b y + c
        # y' = d x + e y + f
        #
        # This is really just a 2D matrix inversion
        #
        # e (x' - c) = ae x + be y
        # b (y' - f) = bd x + be y
        # e x' - b y' + (bf - ec) = (ae - bd) x
        # x = (e x' - b y' + (bf - ec)) / (ae - bd)
        #
        # a (y' - f) = ad x + ae y
        # d (x' - c) = ad x + bd y
        # -d x' + a y' + (dc - af) = (ae - bd) y
        # y = (-d x' + a y' + (dc - af)) / (ae - bd)

        # x = ( e x' - b y' + (bf - ce)) / (ae - bd)
        # y = (-d x' + a y' + (cd - af)) / (ae - bd)

        (a0,b0,c0,d0,e0,f0) = self.scalars
        det_inv = 1. / (a0*e0 - b0*d0)

        a1 =  e0 * det_inv
        b1 = -b0 * det_inv
        c1 =  (b0*f0 - c0*e0) * det_inv

        d1 = -d0 * det_inv
        e1 =  a0 * det_inv
        f1 =  (c0*d0 - a0*f0) * det_inv

        result = Affine(a1, b1, c1, d1, e1, f1)
        result.inverted = self
        self.inverted = result

        return result

    #===========================================================================
    def to_matrix(self):
        """Matrix representation of the transform."""

        return Matrix3.from_scalars(self.a, self.b, self.c,
                                    self.d, self.e, self.f,
                                       0.0,    0.0,    1.0,
                                    recursive=True)

    #===========================================================================
    @staticmethod
    def from_matrix(matrix, tol=1.e-12):
        """Affine representation of the given Matrix.

        Input:
            matrix      matrix to convert to a Affine transform object.
            tol         numeric tolerance to apply to the bottom row of the
                        array, which should always equal (0,0,1). The value is
                        1.e-12 by default. Use None to skip this validation
                        step.
        """

        if tol is not None:
            row2 = matrix.to_vector(0, 2, classes=[], recursive=True)
            (cell0, cell1, cell2) = row.to_scalars()

            if ((cell0.abs() > tol).any() or
                (cell1.abs() > tol).any() or
                ((cell1 - 1.).abs() > tol).any()):
                    raise ValueError('not a valid Affine transformation matrix')

        row0 = matrix.to_vector(0, 0, classes=[], recursive=True)
        (a,b,c) = row.to_scalars()

        row1 = matrix.to_vector(0, 1, classes=[], recursive=True)
        (d,e,f) = row.to_scalars()

        return Affine(a,b,c,d,e,f)

    ############################################################################
    # Methods
    ############################################################################

    def apply(self, pt, recursive=True):
        """Apply the Affine transform to this point."""

        (x,y) = Pair.as_pair(pt, recursive).to_scalars()
        (a,b,c,d,e,f) = self.abcdef[recursive]

        # x' = a x + b y + c
        # y' = d x + e y + f

        x1 = a * x + b * y + c
        y1 = d * x + e * y + f

        return Pair.from_scalars(x1,y1)

    #===========================================================================
    def undo(self, pt, recursive=True):
        """Apply the inverse Affine transform to this point."""

        return self.inverse().apply(pt, recursive)

    ############################################################################
    # Operations
    ############################################################################

    def __mul__(self, arg):
        return Matrix.from_matrix(self.to_matrix() * arg.to_matrix(), tol=None)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Affine(unittest.TestCase):

  def runTest(self):

    # Simple transformation
    T = Affine(0,1,1,1,0,1) # swap x/y and add 1 to each
    p = Pair(np.random.randn(12,13,2))
    (x,y) = p.to_scalars()

    p2 = T.apply(p)
    (x2,y2) = p2.to_scalars()

    dx = x2 - (y + 1)
    dy = y2 - (x + 1)
    self.assertTrue(dx.abs().max() < 1.e-15)
    self.assertTrue(dy.abs().max() < 1.e-15)

    # Simple transformation, derivatives on transformed point
    dp_dvec = Pair(np.random.randn(12,13,2,3), drank=1)
    p.insert_deriv('vec', dp_dvec)
    (x,y) = p.to_scalars()

    p2 = T.apply(p, recursive=True)
    (x2,y2) = p2.to_scalars()

    dx = x2 - (y + 1)
    dy = y2 - (x + 1)
    self.assertTrue(dx.abs().max() < 1.e-15)
    self.assertTrue(dy.abs().max() < 1.e-15)

    self.assertTrue(abs(x2.d_dvec.values - y.d_dvec.values).max() < 1.e-15)
    self.assertTrue(abs(y2.d_dvec.values - x.d_dvec.values).max() < 1.e-15)

    # Multidimensional transformation with derivatives on coefficients
    b = Scalar(np.random.randn(13),
               derivs={'v': Scalar(np.random.randn(13,3), drank=1)})
    d = Scalar(np.random.randn(12,1),
               derivs={'v': Scalar(np.random.randn(13,3), drank=1),
                       't': Scalar(np.random.randn(13), drank=0)})

    T = Affine(0,b,1,d,0,1) # scale x/y and add 1 to each
    self.assertEqual(T.shape, (12,13))

    p = Pair(np.random.randn(12,13,2))
    (x,y) = p.to_scalars()

    p2 = T.apply(p)
    (x2,y2) = p2.to_scalars()

    dx = x2 - (b*y + 1)
    dy = y2 - (d*x + 1)
    self.assertTrue(dx.abs().max() < 1.e-15)
    self.assertTrue(dy.abs().max() < 1.e-15)

    dx_dt_diff = x2.d_dt
    dy_dt_diff = y2.d_dt - x*d.d_dt
    self.assertTrue(dx_dt_diff.abs().max() < 1.e-15)
    self.assertTrue(dy_dt_diff.abs().max() < 1.e-15)

    dx_dv_diff = x2.d_dv - y*b.d_dv
    dy_dv_diff = y2.d_dv - x*d.d_dv
    self.assertTrue(np.all(dx_dv_diff.abs().values < 1.e-15))
    self.assertTrue(np.all(dy_dv_diff.abs().values < 1.e-15))

    # Transformation with derivatives on coefficients and point
    p.insert_deriv('v', Pair(np.random.randn(12,13,2,3), drank=1))
    p.insert_deriv('t', Pair(np.random.randn(12,13,2),   drank=0))
    (x,y) = p.to_scalars()

    p2 = T.apply(p)
    (x2,y2) = p2.to_scalars()

    dx = x2 - (b*y + 1)
    dy = y2 - (d*x + 1)
    self.assertTrue(dx.abs().max() < 1.e-15)
    self.assertTrue(dy.abs().max() < 1.e-15)

    dx_dt_diff = x2.d_dt -  b.wod * y.d_dt
    dy_dt_diff = y2.d_dt - (d.wod * x.d_dt + x.wod * d.d_dt)
    self.assertTrue(dx_dt_diff.abs().max() < 1.e-15)
    self.assertTrue(dy_dt_diff.abs().max() < 1.e-15)

    dx_dv_diff = x2.d_dv - (b.wod * y.d_dv + y.wod * b.d_dv)
    dy_dv_diff = y2.d_dv - (d.wod * x.d_dv + x.wod * d.d_dv)
    self.assertTrue(np.all(dx_dv_diff.abs().values < 1.e-15))
    self.assertTrue(np.all(dy_dv_diff.abs().values < 1.e-15))

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
