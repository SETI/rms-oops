################################################################################
# shapes2d/ellipse.py
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

import numpy as np
from polymath import *
from shape2d  import Shape2D
from conic    import Conic
from affine   import Affine

from point    import Point
from line     import Line, HalfLine, Segment

TWOPI = 2. * np.pi

#*******************************************************************************
# Ellipse
#*******************************************************************************
class Ellipse(Shape2D, Conic):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    An ellipse.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(pt0, rpt, ratio=1.):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Constructor for an Ellipse object, defined by a center, a point on
        the long axis, and an axial ratio.

        Input:
            pt0         center of the ellipse.
            rpt         point on the ellipse's long axis. This corresponds to
                        parameter t = 0 = 2pi.
            ratio       axial ratio b/a.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.pt0 = Pair.as_pair(pt0)
        self.pt1 = Pair.as_pair(rpt)
        self.dx  = self.pt0 - self.pt1
        self.r_sq = self.dx.norm_sq()
        self.r    = self.r_sq.sqrt()

        self.ratio = Scalar.as_scalar(ratio)
        self.ratio_sq = self.ratio**2
        self.rb    = self.r * self.ratio
        self.rb_sq = self.r_sq * self.ratio_sq
        self.dy = self.dx.rot90(recursive=True)

        self.shape = self.r_sq.shape

        #---------------------------------
        # To be used by Arc subclass
        #---------------------------------
        self.fill_limits(Scalar.ZERO, Scalar.TWOPI)

        #---------------------------------------------
        # Fill in Conic section coefficients
        # a x^2 + b xy + c y^2 + d x + e y + f = 0
        #---------------------------------------------
        self.a = Scalar.ONE
        self.c = Scalar.ratio_sq

        #--------------------------------------------------------------------
        # We still need to solve for b, d, e, f
        # Choosing four points on the Ellipse should allow us to do this
        #--------------------------------------------------------------------
        (x0,y0) = self.pt1.to_scalars()
        (x1,y1) = (self.pt0 + self.dy).to_scalars()
        (x2,y2) = (self.pt0 - self.dx).to_scalars()
        (x3,y3) = (self.pt0 - self.dy).to_scalars()

        #-------------------------------------------------------
        # a x^2 + b xy + c y^2 + d x + e y + f = 0
        #
        # b (x0 y0) + d x0 + e y0 + f = -(a x0^2 + c y0^2)
        # b (x1 y1) + d x1 + e y1 + f = -(a x1^2 + c y1^2)
        # b (x2 y2) + d x2 + e y2 + f = -(a x2^2 + c y2^2)
        # b (x3 y3) + d x3 + e y3 + f = -(a x3^2 + c y3^2)
        #
        # Express as a linear least-square problem:
        # A x = B
        # Solve for x = (b, d, e, f)
        #-------------------------------------------------------
        A = Matrix.matrix_from_scalars(x0 * y0, x0, y0, 1.,
                                       x1 * y1, x1, y1, 1.,
                                       x2 * y2, x2, y2, 1.,
                                       x3 * y3, x3, y3, 1.,
                                       recursive=True)
        B = Vector.vector_from_scalars(-self.a * x0*x0 - self.c * y0*y0,
                                       -self.a * x1*x1 - self.c * y1*y1,
                                       -self.a * x2*x2 - self.c * y2*y2,
                                       -self.a * x3*x3 - self.c * y3*y3,
                                       recursive=True)
        x = A.inverse() * B
        (b,d,e,f) = x.to_scalars()
        self.b = b
        self.d = d
        self.e = e
        self.f = f

        self.fill_conic_attributes()
    #===========================================================================



    #===========================================================================
    # fill_limits
    #===========================================================================
    def fill_limits(self, tmin, tmax):
        self.tmin = Scalar.as_scalar(tmin)
        self.tmax = Scalar.as_scalar(tmax)
        self.dt   = (self.tmax - self.tmin) % TWOPI
        self.end0 = self.point_at(self.tmin)
        self.end1 = self.point_at(self.tmax)
    #===========================================================================



    #===========================================================================
    # from_conics
    #===========================================================================
    @staticmethod
    def from_conics(a,b,c,d,e,f):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Construct an Ellipse object from the coefficients of a conic section:
                a x^2 + b xy + c y^2 + d x + e y + f = 0
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------------------------
        # Algorithm stolen from http://mathworld.wolfram.com/Ellipse.html
        # Note alternative scaling on b, d, e used by Wolfram
        #--------------------------------------------------------------------

        #-------------------------
        # Scale so that a = 1.
        #-------------------------
        a = Scalar.as_scalar(a)
        a_inv = 1. / a

        b = Scalar.as_scalar(b) * a_inv * 0.5
        c = Scalar.as_scalar(c) * a_inv
        d = Scalar.as_scalar(d) * a_inv * 0.5
        e = Scalar.as_scalar(e) * a_inv * 0.5
        f = Scalar.as_scalar(f) * a_inv

        a = Scalar.ONE

        discr = b*b - a*c
        if (discr > 0.).any():
            raise ValueError('conic coefficients do not describe an Ellipse')

        discr_inv = 1. / discr

        #------------------------------
        # Formula for the center
        #------------------------------
        x0 = (c*d - b*f) * discr_inv
        y0 = (a*f - b*d) * discr_inv

        #---------------------------------------
        # Formula for the long and short radii
        #---------------------------------------
        c_minus_a = c - a

        term0 = 2 *(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g) * discr_inv
        term1 = -(a + c)
        term2 = (c_minus_a**2 + 4*b**2).sqrt()

        r_long  = (term0 / (term1 - term2)).sqrt()
        r_short = (term0 / (term1 + term2)).sqrt()

        #---------------------------------------------------
        # Formula for the rotation angle of the long axis
        #---------------------------------------------------
        angle = 0.5 * (2*b).arctan2(c_minus_a)

        #---------------------------
        # Construct the Ellipse
        #---------------------------
        pt0 = Pair.from_scalars(x0 y0)
        pt1 = pt0 + r_long * Pair(angle.cos(), angle.sin())
        ratio = r_short/r_long

        return Ellipse(pt0, rpt, ratio)
    #===========================================================================



    ############################################################################
    # Support methods
    ############################################################################

    #===========================================================================
    # _line_intersection_params
    #===========================================================================
    def _line_intersection_params(self, line):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the parameter values where a Line intersects an Ellipse.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return line._line_conic_intersection_params(self)
    #===========================================================================



    #===========================================================================
    # _affine_to_centered_unit_circle
    #===========================================================================
    def _affine_to_centered_unit_circle(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Affine transform object which maps every point on this ellipse to
        a point on a unit circle centered on the origin.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------------------------
        # Apply a shift, rotation, and scaling in that order
        #-------------------------------------------------------

        # Shift so that the center point (x0,y0) lands at the origin
        (x0,y0) = self.pt0.to_scalars()
        shift = Affine(1., 0., -x0,
                       0., 1., -y0)

        # Long axis vector is dpt. Rotate to place dpt on the x-axis
        (dx,dy) = self.dpt.unit().to_scalars()
        rotate = Affine( dx, dy, 0.,
                        -dy, dx, 0.)

        # Radii are (self.r, self.rb
        r_inv = 1. / self.r
        rb_inv = 1. / self.rb
        scale = Affine(r_inv,  r_inv, 0.,
                       rb_inv, rb_inv,  0.)

        # Construct the transform and return
        return scale * rotate * shift
    #===========================================================================



    #===========================================================================
    # _affine_to_centered_xy_ellipse
    #===========================================================================
    def _affine_to_centered_xy_ellipse(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Affine transform object which maps every point on this ellipse to
        a point on an ellipse of the same shape but centered at the origin and
        with the long axis along the x-axis.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-----------------------------------------------
        # Apply a shift and rotation in that order
        #-----------------------------------------------

        #--------------------------------------------------------------
        # Shift so that the center point (x0,y0) lands at the origin
        #--------------------------------------------------------------
        (x0,y0) = self.pt0.to_scalars()
        shift = Affine(1., 0., -x0,
                       0., 1., -y0)

        #--------------------------------------------------------------
        # Long axis vector is dpt. Rotate to place dpt on the x-axis
        #--------------------------------------------------------------
        (dx,dy) = self.dpt.unit().to_scalars()
        rotate = Affine( dx, dy, 0.,
                        -dy, dx, 0.)

        #---------------------------------------
        # Construct the transform and return
        #---------------------------------------
        return rotate * shift
    #===========================================================================



    #===========================================================================
    # _conic_intersection_params
    #===========================================================================
    def _conic_intersection_params(conic, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A tuple of the four parameter values where this ellipse intersects a
        given conic.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------------
        # Transform the coordinate system so this is a unit circle
        #------------------------------------------------------------
        affine = self._affine_to_centered_unit_circle()
        conic2 = conic.apply_affine(affine)

        #--------------------------------
        # Solve for the intersections
        #--------------------------------
        return conic2._unit_circle_intersection_angles(recursive)
    #===========================================================================



    ############################################################################
    # Methods overridden by Arc subclass
    ############################################################################

    #===========================================================================
    # _mask
    #===========================================================================
    def _mask(self, t, obj=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Masks where Scalar t is outside the allowed range.

        Input:
            t       parameter of shape.
            obj     the object to mask and return; if None, then t is masked and
                    returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if obj is not None:
            return obj

        return t
    #===========================================================================



    #===========================================================================
    # _clip
    #===========================================================================
    def _clip(self, t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Clips Scalar t to the allowed range. No masking is applied.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return t
    #===========================================================================



    ############################################################################
    # Methods defined for all classes
    ############################################################################

    #===========================================================================
    # dimensions
    #===========================================================================
    def dimensions(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Scalar dimension of this object: 0 for a point; 1 for a line; 2
        for a shape object that has nonzero area.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (self.r <= Shape2D.PREC).any():
            return (self.r != 0).as_int()

        return Scalar.TWO
    #===========================================================================



    #===========================================================================
    # is_convex
    #===========================================================================
    def is_convex(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Boolean True if the shape is convex.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Boolean.TRUE
    #===========================================================================



    #===========================================================================
    # point_at
    #===========================================================================
    def point_at(t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Parameterization of shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        t = Scalar.as_scalar(t)
        return (self.pt0 + self.dx * t.cos() + self.dy * t.sin())
    #===========================================================================



    #===========================================================================
    # param_at
    #===========================================================================
    def param_at(pt, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Parameter at a point, which is assumed to fall on the edge of this
        object.

        What happens when the point does not fall on the shape is undetermined.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        diff = pt - self.pt0
        x = self.dx.dot(diff)
        y = self.dy.dot(diff)
        t = y.arctan2(x) % TWOPI

        if mask:
            return self._mask(t)
        else:
            return t
    #===========================================================================



    #===========================================================================
    # param_limits
    #===========================================================================
    def param_limits(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Parameter limits to define the shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return (self.tmin, self.tmax)
    #===========================================================================



    #===========================================================================
    # closest
    #===========================================================================
    def closest(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Tuple containing the pairs of closest points between the edges of
        this object and the given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         (self_pt, arg_pt)
            self_pt     the point on this shape that falls closest to any point
                        on the given Shape2D object.
            arg_pt      the associated closest point on the given Shape2D
                        object.

        Note: If multiple pairs of points are separated by the exact same
        distance, only one result will be returned. The returned pairs of points
        are guaranteed to be unmasked where the shapes are initially unmasked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------------------
        # Transform to a frame where the ellipse is centered and the semimajor
        # axis is parallel to x. After this transformation, angles between
        # vectors are unchanged. Dot products are preserved.
        #
        # pt0 = (x0,y0)
        # pt1 = (x1,y1)
        #
        # dot(pt0,pt1) = (x0 * x1 + y0 * y1)
        #
        # Now transform to convert the ellipse to a unit circle
        #   (x',y') = (x/self.r, y/self.rb)
        # or
        #   (x,y) = (self.r * x', self.rb * y')
        #
        # dot(pt0', pt1') = (self.r_sq * x0' * x1' + self.rb_sq * y0' * y1')
        #       = self.r_sq * (x0' * x1' + self.ratio_sq * y0' + y1')
        #
        # Conclusion: If we work in the frame of the centered unit circle, our
        # geometry formulas will work provided we re-define the dot product as
        # shown above.
        #
        # dot(pt0', dpt') = 0 for dpt' as the local normal.
        #
        #   x0' * dx' + self.ratio_sq * y0' + dy' = 0
        #
        # Let tangent (dx',dy') = (-self.ratio_sq * y0', x0')
        #
        # Therefore normal = (x0', self.ratio_sq * y0')
        #----------------------------------------------------------------------

        #-------------------------------
        # Closest, Ellipse to Point
        #-------------------------------
        if type(arg) in (Point, Pair):

            # Transform to the centered unit circle
            affine = self._affine_to_centered_unit_circle()
            pt = affine.apply(arg)
            (px,py) = pt.to_scalars()

            # ept = (x,y) is a point on the transformed ellipse.
            #
            # The local tangent is (-y,x).
            #
            # Solve for ept such that
            #   dot(ept - pt, tangent(ept)) = 0
            # and
            #   |ept| = 1
            #
            # (x - px) * ((-y) + self.ratio_sq * (y - py) * x = 0
            #
            # Let Q = self.ratio_sq
            #
            # (Q-1) x y - Q py x + px y = 0
            #
            # Eliminate x:
            #   y px = x [(1-Q) y + Q py]
            #   y^2 px^2 = x^2 [(1-Q) y + Q py]^2
            #   y^2 px^2 = (1 - y^2) [(1-Q) y + Q py]^2
            #
            # Solve for y:
            #   y^2 px^2 + (y^2 - 1) [(1-Q) y + Q py]^2 = 0
            #
            # This is a quartic equation:
            #   p4 = (1-Q)^2
            #   p3 = 2 Q (1-Q) py
            #   p2 = px^2 + Q^2 py^2 - (1-Q)^2
            #   p1 = -2 Q (1-Q) py
            #   p0 = -Q^2 py^2
            Q = self.ratio_sq
            one_minus_Q = 1. - Q
            one_minus_Q_sq = one_minus_Q**2
            Q_py = Q * py
            py_term = 2. * one_minus_Q * Q_py
            Q_py_sq = Q_py**2
            px_sq = px**2

            p4 = one_minus_Q_sq
            p3 = py_term
            p2 = px_sq + Q_py_sq - one_minus_Q_sq
            p1 = -py_term
            p0 = -Q_py_sq

            poly = Vector.from_scalars(p4,p3,p2,p1,p0, classes=(Polynomial,))
            roots = poly.roots()
            y0 = roots.mask_where(root.abs() > 1.)
            x0 = (one_minus_Q * y + Q_py) / px

            # The above procedure fails if px = 0, so we also need to try the
            # alternative method by eliminating y first.
            #   (Q-1) x y - Q py x + px y = 0
            #
            #   x (Q py) = y [(Q-1) x + px]
            #
            #   x^2 (Q py)^2 = y^2 [(Q-1) x + px]^2
            #   x^2 (Q py)^2 = (1 - x^2) [(Q-1) x + px]^2
            #
            # Solve for x:
            #   x^2 (Q py)^2 + (1 - x^2) [(1-Q) x - px]^2 = 0
            #
            # This is a quartic equation:
            #   p4 = -(1-Q)^2
            #   p3 = 2 (1-Q) px
            #   p2 = (Q py)^2 + px^2 - (1-Q)^2
            #   p1 = -2 (1-Q) px
            #   p0 = -px^2

            px_term = 2. * one_minus_Q * px

            p4 = -one_minus_Q_sq
            p3 = px_term
            # p2 = same as above
            p1 = -px_term
            p0 = -px_sq

            poly = Vector.from_scalars(p4,p3,p2,p1,p0, classes=(Polynomial,))
            roots = poly.roots()
            x1 = roots.mask_where(root.abs() > 1.)
            y1 = (px - one_minus_Q * x) / Q_py

            x = x0 + x1
            y = y0 + y1
            circle_pts = [Pair.from_scalars(x[k],y[k]) for k in range(8)]
            self_pts = [affine.undo(cpt) for cpt in circle_pts]
            points = 8*[arg]

            # Also check endpoints of Arcs
            if type(self).__name__ == 'Arc':
                self_pts += [self.end0, self.end1]
                points   += 2*[arg]

            return Shape2D._closest_of_pairings(self_pts, points)

        #-----------------------------
        # Closest, Ellipse to Line
        #-----------------------------
        if isinstance(self, Line):
            line = arg

            # Find intersections
            (s0, s1) = self._line_intersection_params(line)
            self_pts = [line.point_at(s0), line.point_at(s1)]
            line_pts = list(self_pts)   # a copy

            # Transform line to centered unit circle coordinates
            affine = self._affine_to_centered_unit_circle()
            line_pt0 = affine.apply(line.pt0)
            line_pt1 = affine.apply(line.pt1)
            newline = type(line).__init__(line_pt0, line_pt1)

            # The point on an ellipse closest to a line is the point where the
            # tangent and the line are parallel
            #
            # Affine transformations preserve parallel lines, so this condition
            # must be satisfied in the transformed coordinates too.
            #
            # An ellipse point ept = (x,y). The local tangent is (-y,x).
            #
            # Solve:
            #   (-y,x) = constant * line.dpt
            # and
            #   x^2 + y^2 = 1
            #
            # Let line.dpt = (dx,dy)
            #
            # (-y,x) = constant * (dx,dy)
            # -y = k * dx;
            #  x = k * dy
            #
            # k^2 * (dx^2 + dy^2) = 1
            # k = +/- 1. / sqrt(dx^2 + dy^2)
            #
            # x =  k * dy
            # y = -k * dx

            # Two solutions exist: (cx,cy) and its negative
            k = newline.r_inv
            (dx,dy) = newline.dpt.to_scalars()
            cx = -k*y
            cy =  k*x
            circle_pt = Pair.from_scalars(cx,cy)
            ellipse_pt0 = affine.undo( circle_pt)
            ellipse_pt1 = affine.unto(-circle_pt)

            # This is the point on the ellipse. Where is the point on the line?
            #   line(t) = pt0 + t * dpt
            #   pt0 = (x0,y0)
            #
            #   dot(line(t) - ept, dpt) = 0
            #   (x0 - x + t*dx)*dx + self.ratio_sq * (y0 - y + t*dy)*dy = 0
            #
            # t * (dx^2 + dy^2) + (x0-x)*dx + self.ratio_sq * (y0-y)*dy = 0
            # t = -((x0-x)*dx + self.ratio_sq * (y0-y)*dy) / (dx^2 + dy^2)

            (x0,y0) = newline.pt0.to_scalars()

            t0 = -((x0-cx)*dx + self.ratio_sq * (y0-cy)*dy) * newline.r_sq_inv
            t1 = -((x0+cx)*dx + self.ratio_sq * (y0+cy)*dy) * newline.r_sq_inv

            self_pts += [ellipse_pt0,       ellipse_pt1      ]
            line_pts += [line.point_at(t0), line.point_at(t1)]

            # Also consider endpoints of line
            if type(line) in (HalfLine, Segment):
                (self_pt, line_pt) = self.closest(line.pt0)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(line) == Segment:
                (self_pt, line_pt) = self.closest(line.pt1)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            # Also check endpoints of Arcs
            if type(self).__name__ == 'Arc':
                (line_pt0, self_pt0) = line._closest(self.end0)
                (line_pt1, self_pt1) = line._closest(self.end1)
                self_pts += [self_pt0, self_pt1]
                line_pts += [line_pt0, line_pt1]

            # Select and return closest pairings
            return Shape2D._closest_of_pairings(self_pts, line_pts)

        #---------------------------------
        # Closest, Ellipse to Ellipse
        #---------------------------------
        if isinstance(self, Ellipse):
            ellipse = arg

            # Find intersections
            (t0,t1,t2,t3) = self._conic_intersection_params(ellipse)
            self_pts = [self.point_at(t0), self.point_at(t1),
                        self.point_at(t2), self.point_at(t3)]
            ellipse_pts = list(self_pts)   # a copy

            # Find ellipse parameters t and s that satisfy
            #   dot(e0(t).slope2d(), e0(t) - e1(s)) = 0
            # where e1(s) is the closest point on the second ellipse to e0(t)
            e0 = type(self).__init__(self.pt0.wod, self.pt1.wod, self.ratio.wod)
            e0.fill_limits(self.tmin.wod, self.tmax.wod)

            e1 = type(self).__init__(ellipse.pt0.wod, ellipse.pt1.wod,
                                     ellipse.ratio.wod)
            e1.fill_limits(self.tmin.wod, self.tmax.wod)

            # Initial guess
            t = e0.param_at(e1.pt0)
            t.insert_deriv('t', 1.)     # insert self-derivative

            # Apply Newton's method
            # t' = t - f(t) / f'(t)
            for iter in range(ITERS):
                ept = e0.point_at(t)
                f = e0.slope2d(ept).dot(ept - e1.closest(ept))  # recursive call
                t = t - f/f.d_dt

            # Restore derivatives to t
            # df_dx = df_dt * dt_dx
            # dt_dx = df_dx / df_dt
            df_dt = f.d_dt
            ept = self.point_at(t.wod)
            e1pt = ellipse.closest(ept)
            f = self.slope2d(ept).dot(ept - e1pt)

            t_derivs = {}
            for (key, df_dkey) in f.derivs.iteritems():
                t_derivs[key] = df_dkey / df_dt

            t = t.insert_derivs(t_derivs)
            self_pt = self.point_at(t)

            self_pts.append(self_pt)
            ellipse_pts.append(e1pt)

            # Also check endpoints of Arcs
            if type(self).__name__ == 'Arc':
                (ellipse_pt0, self_pt0) = ellipse._closest(self.end0)
                (ellipse_pt1, self_pt1) = ellipse._closest(self.end1)
                self_pts    += [self_pt0,    self_pt1]
                ellipse_pts += [ellipse_pt0, ellipse_pt1]

            if type(ellipse).__name__ == 'Arc':
                (self_pt0, ellipse_pt0) = self._closest(ellipse.end0)
                (self_pt1, ellipse_pt1) = self._closest(ellipse.end1)
                self_pts    += [self_pt0,    self_pt1]
                ellipse_pts += [ellipse_pt0, ellipse_pt1]

            # Select and return closest pairings
            return Shape2D._closest_of_pairings(self_pts, ellipse_pts)

        #---------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #---------------------------------------------------------------
        return arg.closest(self)[::-1]
    #===========================================================================



    #===========================================================================
    # intersections
    #===========================================================================
    def intersections(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Points defining intersections between the edges of this shape and the
        given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         all the Points that fall on the intersections of these
                        two Shape2D objects. The array shapes of self and arg
                        are broadcasted together and the returned result is a
                        Point with one extra leading dimension, equal to the
                        maximum number of possible intersections. Intersections
                        will be masked where the shape edges do not intersect or
                        are duplicated.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------
        # Intersections, Ellipse to Point
        #-------------------------------------
        if type(arg) == Point:
            pt = arg
            pt = pt.mask_where(self.eval_conic(pt).abs() <= Shape2D.PREC)
            return Point(pt.reshape((1,) + pt.shape))

        #------------------------------------
        # Intersections, Ellipse to Line
        #------------------------------------
        if isinstance(arg, Line):
            line = arg

            (s0, s1) = line._line_conic_intersection_params(self)
            pt0 = arg.point_at(s0)
            pt1 = arg.point_at(s1)

            return Point(Qube.stack(pt0, pt1))

        #-------------------------------------
        # Intersections, Ellipse to Ellipse
        #-------------------------------------
        if isinstance(arg, Ellipse):
            ellipse = arg

            (t0,t1,t2,t3) = self._conic_intersection_params(ellipse)
            return Point(Qube.stack(self.point_at(t0), self.point_at(t1),
                                    self.point_at(t2), self.point_at(t2)))

        #----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #----------------------------------------------------------------
        return arg.intersections(self)
    #===========================================================================



    #===========================================================================
    # tangents_from
    #===========================================================================
    def tangents_from(self, pt):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The two points where this Shape2D object is tangent to a line from
        the given Point.

        Note: If the two points are degenerate, the second one is masked.

        Input:
            self        this shape.
            pt          a Point or Pair from which to draw the tangent lines.

        Return:         the two Points on the edge of this shape that are
                        tangent to a line from the given point. The array shapes
                        of self and pt are broadcasted together and the returned
                        result is a tuple containing two Scalar objects with
                        this shape. Tangent points are be masked if they do not
                        exist or are duplicated.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------------------
        # Solve for the points where
        # e(t) - pt || tangent(e(t)
        #
        # This equation is satisfied after affine transformation because
        # tangents remain tangents.
        #------------------------------------------------------------------
        affine = self._affine_to_centered_unit_circle()
        pt = affine.apply(arg)

        #----------------------------------------
        # Solve for points tangent to circle
        #----------------------------------------
        circle = Circle(Pair.ZEROS, Pair.XAXIS)
        (pt0, pt1) = circle.tangents_from(pt)

        pt0 = affine.undo(pt0)
        pt1 = affine.undo(pt1)

        #-------------------------------
        # Reapply masks if necessary
        #-------------------------------
        t0 = self.param_at(pt0)
        t1 = self.param_at(pt1)

        return (self.point_at(t0), self.point_at(t1))
    #===========================================================================



    #===========================================================================
    # tangent_at
    #===========================================================================
    def tangent_at(self, t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Line object tangent to this Shape2D object at the given parameter
        value.

        Input:
            self        this shape.
            t           a parameter value on this shape.

        Return:         a Line object defining tangent points. The array shapes
                        of self and t are broadcasted together and the returned
                        result is a Line object with this shape. Tangent lines
                        will be masked if the tangent value is undefined.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pt0 = self.point_at(t)

        (dx,dy) = self.slope2d()
        dpt = Pair.from_scalars(dx,dy)

        return Line(pt0, pt0 + dpt)
    #===========================================================================



    #===========================================================================
    # normal_at
    #===========================================================================
    def normal_at(self, t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The outward HalfLine object normal to this Shape2D object at the
        given parameter value.

        Input:
            self        this shape.
            t           a parameter value on this shape.

        Return:         a HalfLine object defining normal points. The array
                        shapes of self and t are broadcasted together and the
                        returned result is a Line object with this shape.
                        Tangent lines will be masked if the tangent value is
                        undefined.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pt0 = self.point_at(t)
        dpt = self.perp2d(pt0)
        return HalfLine(pt0, pt0 + dpt)
    #===========================================================================



    #===========================================================================
    # is_subset_of
    #===========================================================================
    def is_subset_of(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object is as subset of (i.e., is entirely contained by)
        the given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a subset of the given
                        shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------
        # Is subset of, Ellipse to Point
        #----------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            return (pt - self.pt0).norm() - self.r <= Shape2D.PREC)

        #----------------------------------
        # Is subset of, Ellipse to Line
        #----------------------------------
        if isinstance(type(arg), Line):
            line = arg
            return self.pt0.is_subset_of(line) & (self.r <= Shape2D.PREC)

        #-------------------------------------
        # Is subset of, Ellipse to Ellipse
        #-------------------------------------
        if isinstance(type(arg), Ellipse):
            affine = self._affine_to_centered_unit_circle()
            ellipse = arg.apply_affine(affine, shapeclass=Ellipse)
            innermost = ellipse.closest(Pair.ZEROS)
            return (innermost.norm_sq() >= 1. &
                    ellipse.is_superset_of(Pair.ZEROS))

        #--------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #--------------------------------------------------------------
        return arg.is_superset_of(self)
    #===========================================================================



    #===========================================================================
    # is_superset_of
    #===========================================================================
    def is_superset_of(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object is as superset of (i.e., entirely contains) the
        given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a superset of the given
                        shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------
        # Is superset of, Ellipse to Point
        #-------------------------------------
        if type(arg) in (Point, Pair):
            affine = self._affine_to_centered_unit_circle()
            pt = affine.apply(pt)
            return pt.norm_sq() <= 1.

        #-----------------------------------------------
        # Is superset of, Ellipse to Line or HalfLine
        #-----------------------------------------------
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        #---------------------------------------
        # Is superset of, Ellipse to Segment
        #---------------------------------------
        if isinstance(type(arg), Segment):
            line = arg
            affine = self._affine_to_centered_unit_circle()
            inside0 = affine.apply(line.pt0).norm_sq() <= 1.
            inside1 = affine.apply(line.pt1).norm_sq() <= 1.
            return inside0 & inside1

        #---------------------------------------
        # Is superset of, Ellipse to Ellipse
        #---------------------------------------
        if isinstance(type(arg), Ellipse):
            ellipse = arg
            return ellipse.is_subset_of(self)

        #--------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #--------------------------------------------------------------
        return arg.is_superset_of(self)
    #===========================================================================



    #===========================================================================
    # is_disjoint_from
    #===========================================================================
    def is_disjoint_from(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the this object and the given Shape2D object are disjoint
        (i.e., do not touch or overlap).

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is disjoing from the given
                        shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------
        # Is disjoint from, Ellipse to Point
        #---------------------------------------
        if type(arg) in (Point, Pair):
            affine = self._affine_to_centered_unit_circle()
            pt = affine.apply(pt)
            return pt.norm_sq() > 1.

        #--------------------------------------
        # Is disjoint from, Ellipse to Line
        #--------------------------------------
        if isinstance(type(arg), Line):
            affine = self._affine_to_centered_unit_circle()
            line = Line(affine.apply(line.pt0), affine.apply(line.pt1))
            line_pt = line.closest(Pair.ZEROS)
            return line_pt.norm_sq() > 1.

        #----------------------------------------
        # Is disjoint from, Ellipse to Ellipse
        #----------------------------------------
        if isinstance(type(arg), Ellipse):
            affine = self._affine_to_centered_unit_circle()
            ellipse = arg.apply_affine(affine, shapeclass=Ellipse)
            innermost = ellipse.closest(Pair.ZEROS)
            return (innermost.norm_sq() > 1. &
                    ~ellipse.is_superset_of(Pair.ZEROS))

        #--------------------------------------
        # Otherwise use the general method
        #--------------------------------------
        return super(Shape2D, self).is_disjoint_from(arg)
    #===========================================================================



    #===========================================================================
    # touches
    #===========================================================================
    def touches(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the this object and the given Shape2D touch but do not share
        any common interior points.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if the shapes touch but share no common
                        interior points.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------
        # Touches, Ellipse to Point
        #------------------------------
        if type(arg) in (Point, Pair):
            affine = self._affine_to_centered_unit_circle()
            pt = affine.apply(arg)
            return (((pt.norm_sq() - 1.).abs() < 2.*Shape2D.PREC) &
                    self.param_at(pt).antimask)

        #-----------------------------
        # Touches, Ellipse to Line
        #-----------------------------
        if isinstance(type(arg), Line):
            affine = self._affine_to_centered_unit_circle()
            line = Line(affine.apply(arg.pt0), affine.apply(arg.pt1))
            closest = line.closest(Pair.ZEROS)[0]

            # Touches if line's closest point to origin is on unit circle and
            # circle is unmasked here
            touches = (((closest.norm_sq() - 1.).abs() <= 2.*Shape2D.PREC) &
                        self.param_at(closest).antimask)
            if type(arg) == Line: return touches

            if type(self).__name__ == 'Arc':
                touches |= (self.end0.is_subset_of(line) |
                            self.end1.is_subset_of(line))

            touches2 = (line.pt0.norm_sq() - 1.).abs() <= 2.*Shape2D.PREC
            if type(self).__name__ != 'Arc':
                touches2 &= line.pt0.dot(line.dpt) >= 1.

            if type(arg) == HalfLine: return touches | touches2

            touches3 = (line.pt1.norm_sq() - 1.).abs() <= 2.*Shape2D.PREC
            if type(self).__name__ != 'Arc':
                touches3 &= line.pt1.dot(line.dpt) <= 1.

            return touches | touches2 | touches3

        #--------------------------------------
        # Otherwise use the general method
        #--------------------------------------
        return super(Shape2D, self).touches(arg)
    #===========================================================================


#*******************************************************************************



################################################################################
