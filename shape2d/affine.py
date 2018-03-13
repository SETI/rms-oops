################################################################################
# shape2d/affine.py: Affine transform class and support functions
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division
import numpy as np

class Affine(object):
    """An class that describes 2-D affine transformations. These are transforms
    of the form:
        x' = a x + b y + c
        y' = d x + e y + f
    """

    def __init__(self, a, b, c, d, e, f):
        """Constructor for an Affine transform."""

        self.a = Scalar.as_scalar(self.a)
        self.b = Scalar.as_scalar(self.b)
        self.c = Scalar.as_scalar(self.c)
        self.d = Scalar.as_scalar(self.d)
        self.e = Scalar.as_scalar(self.e)
        self.f = Scalar.as_scalar(self.f)

        # Indexed by recursive = True/False
        self.scalars = (self.a, self.b. self.c, self.d, self.d, self.f)
        self.scalars_wod = (self.a.without_derivs(),
                            self.b.without_derivs(),
                            self.c.without_derivs(),
                            self.d.without_derivs(),
                            self.e.without_derivs(),
                            self.f.without_derivs())
    
        self.abcdef[self.abcdef_wod, self.abcdef]
        self.affine_shape = np.broadcasted_shape(*self.scalars)

        self.inverted = None        # Use to track the inverse transform

    def inverse(self):
        """Return the Affine transform that undoes this one."""

        if self.inverted: return self.inverted

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

    def to_matrix(self):
        """Matrix representation of the transform."""

        return Matrix.from_scalars(self.a, self.b, self.c,
                                   self.d, self.e, self.f,
                                      0.0,    0.0,    1.0,
                                   recursive=True)

    @staticmethod
    def from_matrix(matrix, tol=1.e-12):
        """Affine representation of the given Matrix.

        Input:
            matrix      matrix to convert to a Affine transform object.
            tol         numeric tolerance to apply to the bottom row of the
                        array, e.g., 1.e-12. Use None to skip this validation
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

    def undo(self, pt, recursive=True):
        """Apply the inverse Affine transform to this point."""

        return self.inverse().apply(pt, recursive)

    def intersections_with_unit_circle(self, recursive=True):
        
    ############################################################################
    # Operations
    ############################################################################

    def __mul__(self, arg):
        return Matrix.from_matrix(self.to_matrix() * arg.to_matrix(), tol=None)

################################################################################
