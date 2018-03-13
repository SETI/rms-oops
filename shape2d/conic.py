################################################################################
# shape2d/conics.py: Conic shape class and support functions
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division
import numpy as np

from polymath import *
from affine import Affine

TWOPI = 2. * np.pi

class Conic(object):
    """An abstract class with six Scalar attributes a,b,c,d,e,f, which represent
    an equation:
        a ^2 + b xy + c y^2 + d e + e y + f = 0.
    """

    def __init__(self, a, b, c, d, e, f):
        """Constructor for a Conic to use in those situations where it is not
        used as attributes of another object."""

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

        self.fill_conic_attributes()

    def fill_conic_attributes(self):
        """Fill in additional attributes to support quick calculations of conic
        sections. The six Scalar attributes a,b,c,d,e,f must already be filled
        in."""

        self.a = Scalar.as_scalar(self.a)
        self.b = Scalar.as_scalar(self.b)
        self.c = Scalar.as_scalar(self.c)
        self.d = Scalar.as_scalar(self.d)
        self.e = Scalar.as_scalar(self.e)
        self.f = Scalar.as_scalar(self.f)

        self.scalars= (self.a, self.b. self.c, self.d, self.d, self.f)
        self.scalars_wod = (self.a.without_derivs(),
                            self.b.without_derivs(),
                            self.c.without_derivs(),
                            self.d.without_derivs(),
                            self.e.without_derivs(),
                            self.f.without_derivs())

        # Conic coefficients indexed by recursive = True/False
        self.abcdef[self.abcdef_wod, self.abcdef]

    ############################################################################
    # Transformations of Conics
    ############################################################################

    def swapxy(self):
        """A shallow clone of this Conic with the x and y axes reversed."""

        return Conic(self.c, self.b, self.a, self.e, self.d, self.f)

    def apply_affine(self, affine, recursive=True, shapeclass=None):
        """Apply the given affine transformation to all the points comprising
        this conic.

        Input:
            affine      Affine transformation object to apply.
            recursive   True to include derivatives in transform.
            shapeclass  Shape2D subclass to return. If None (the default), then
                        a Conic object is returned.
        """

        # a x^2 + b xy + c y^2 + d x + e y + f = 0
        #
        # x' = A x + B y + C
        # y' = D x + E y + F

        (a,b,c,d,e,f) = self.abcdef[recursive]
        (A,B,C,D,E,F) = affine.abcdef[recursive]

        a1 = a *   A**2 + b *  A*D        + c *   D**2
        b1 = a * 2*A*B  + b * (A*E + B*D) + c * 2*D*E
        c1 = a *   B**2 + b *  B*E        + c *   E**2
        d1 = a * 2*A*C  + b * (A*F + C*D) + c * 2*D*F  + d * A + e * D
        e1 = a * 2*B*C  + b * (B*F + C*E) + c * 2*B*C  + d * B + e * E
        f1 = a *   C**2 + b *  C*F        + c *   F**2 + d * C + e * F + f

        if shapeclass is None:
            return Conic(a1, b1, c1, d1, e1, f1, g1)

        # Call the from_conics static method of the selected class
        return shapeclass.__dict__['from_conics'].__call__(a1,b1,c1,d1,e1,f1)

    def undo_affine(affine, recursive=True, shapeclass=None):
        """Apply the inverse of the given affine transformation to all the
        points comprising this conic.
        """

        return self.apply_affine(affine.inverse(), recursive, shapeclass)

    ############################################################################
    # Methods
    ############################################################################

    def eval_conic(self, pt, recursive=True):
        """Evaluate the conic at Pair(x,y).

        Inputs:
            pt          Pair at which to evaluate the Conic. The value will be
                        zero if the point falls on the curve.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and the
                        (x,y) pair are broadcasted together.
        """

        (x,y) = Pair.as_pair(pt, recursive).to_scalars()
        (a,b,c,d,e,f) = self.abcdef[recursive]

        return a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    def perp2d(self, pt, recursive=True):
        """Two scalars (dx,dy) that point in the direction of the outward local
        normal to the curve.

        Inputs:
            pt          Pair at which to evaluate the perpendicular.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and the
                        (x,y) pair are broadcasted together.
        """

        (x,y) = Pair.as_pair(pt, recursive).to_scalars()
        (a,b,c,d,e,f) = self.abcdef[recursive]

        # a x^2 + b xy + c y^2 + d x + e y + f = 0
        # 2a x dx + b x dy + b y dx + 2c y dy + d dx + e dy = 0
        #
        # dx (2a x + b y + d) + dy (b x + 2c y + e) = 0
        #
        # Solution is:
        #   dx =  (2c y + b x + e)
        #   dy = -(2a x + b y + d)
        #
        # Tangent direction is (dx,dy)
        # Normal direction is (-dy,dx)

        neg_dy = 2.*a*x + b*y + d
        pos_dx = 2.*c*y + b*x + e

        sign = (x * pos_dx + y * neg_dy).sign()
        return (pos_dx*sign, neg_dy*sign)

    def slope2d(self, pt, recursive=True):
        """Two scalars (dy,dx) that represent the slope of the local tangent.

        Inputs:
            pt          Pair at which to evaluate the slope.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and the
                        (x,y) pair are broadcasted together.
        """

        (pos_dx, neg_dy) = self.perp2d(pt, recursive)
        return (-neg_dy, pos_dx)

    def slope(self, pt, recursive=True):
        """Slope of the local tangent.

        Inputs:
            pt          Pair at which to evaluate the slope.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and the
                        (x,y) pair are broadcasted together.
        """

        (dy,dx) = self.slope2d(pt, recursive)
        return dy/dx

    def perp(self, pt, recursive=True):
        """Slope of the local perpendicular.

        Inputs:
            pt          Pair at which to evaluate the perpendicular.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and the
                        (x,y) pair are broadcasted together.
        """

        (dy,dx) = self.perp2d(pt, recursive)
        return dy/dx

    def _unit_circle_intersection_angles(self, recursive=True):
        """Tuple of the four parameters on a unit circle centered on the origin
        that intersect this Conic.

        Inputs:
            recursive   True to evaluate derivatives at the intersections.

        Return:         A tuple of four Scalars, each with the same shape as the
                        conic, defining the angle (clockwise from the x-axis) of
                        a point on the unit circle that intersects this conic.
                        Points that either do not exist or are duplicates are
                        masked.
        """

        (a0,c0,b0,d0,e0,f0) = self.abcdef[recursive]

        # a x2 + b xy + c y2 + d x + e y + f = 0
        #   x2        +   y2             - 1 = 0
        #
        # Solve for x in terms of y:
        # a x2 + b xy + c y2 + d x + e y + f = 0
        # a x2        + a y2             - a = 0
        #
        # Subtract
        # b xy + (c-a) y2 + d x + e y + (f+a) = 0
        #
        # Let
        #   c' = c - a
        #   f' = f + a
        #
        # x = -(c' y2 + e y + f') / (b y + d)
        #
        # Now solve for y by substituting x into the unit circle equation
        #
        # x2 (b y + d)2 + y2 (b y + d)2 - (b y + d)2 = 0
        #
        # (c' y2 + e y + f')2 + (b y + d)2 (y2 - 1) = 0
        #
        # This is a quartic polynomial.

        c1 = c0 - a0
        f1 = f0 + a0

        # Determine coefficients of the quartic to solve for y
        b0_sq = b0 * b0
        d0_sq = d0 * d0
        b0_d0 = b0 * d0

        p4 = c1**2 + b0_sq
        p3 = 2*(c1*e0 + b0_d0)
        p2 = 2*c0*f1 + e0**2 - b0_sq + d0_sq
        p1 = 2*(e0*f1 - b0_d0)
        p0 = f1**2 - d0_sq

        # Solve the quartic
        poly = Vector.from_scalars(p4, p3, p2, p1, p0, classes=(Polynomial,))
        roots = poly.roots(recursive)

        # Plug y-roots back into the expression for x
        #   x = -(c' y2 + e y + f') / (b y + d)

        angles = []
        for k in range(4):
            y = roots[k]
            x = -((c1*y + e0)*y + f1) / (b0*y + d0)
            angle = y.arctan2(x) % TWOPI
            params.append(angle)

        angles = Qube.stack(*angles).sort()
        return (angles[0], angles[1], angles[2], angles[3])

################################################################################
