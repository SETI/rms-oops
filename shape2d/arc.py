################################################################################
# shapes2d/arc.py
################################################################################

import numpy as np
from polymath import *
from shape2d  import Shape2D
from point    import Point
from line     import Line, HalfLine, Segment
from circle   import Circle

TWOPI = 2. * np.pi

class Arc(Ellipse):
    """A sector of an Ellipse."""

    def __init__(pt0, rpt, ratio=1., tmin=0., tmax=TWPI):
        """Constructor for a Arc object.

        Input:
            pt0         center of the ellipse.
            rpt         point on the ellipse's long axis. This corresponds to
                        parameter t = 0 = 2pi.
            ratio       axial ratio b/a.
            tmin        minimum parameter, measured from the long axis.
            tmax        maximum parameter, measured from the long axis. For
                        arcs that cross t=0, tmax will be less than tmin.
        """

        super(Ellipse, self).__init__()
        self.fill_limits(tmin, tmax)

    ############################################################################
    # Methods overridden by Arc subclass
    ############################################################################

    def _mask(self, t, obj=None):
        """Masks where Scalar t is outside the allowed range.

        Input:
            t       parameter of shape.
            obj     the object to mask and return; if None, then t is masked and
                    returned.
        """

        test = (Scalar.as_scalar(t) - self.tmin) % TWOPI

        if obj is None:
            obj = self

        return Scalar.as_scalar(t).mask_where(test > self.dt)

    ############################################################################
    # Methods defined for all classes
    ############################################################################

    def dimensions(self):
        """The Scalar dimension of this object: 0 for a point; 1 for a line; 2
        for a shape object that has nonzero area."""

        dims = (self.dt > Shape2D.PREC) & (self.r > Shape2D.PREC)
        dims = dims.as_int()

        if (dims == 0).any():
            return dims
        else:
            return Scalar.ONE

    def is_convex(self):
        """Boolean True if the shape is convex."""

        return (self.dimensions() == 0) |
               ((self.dt - TWOPI).abs() <= Shape2D.PREC)

    def is_subset_of(self, arg):
        """True if this object is as subset of (i.e., is entirely contained by)
        the given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a subset of the given
                        shape.
        """

        # Is subset of, Arc to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            # Only possible if the arc radius = 0 or the arc length = 0
            return ((self.pt0.is_subset_of(pt) & self.r <= Shape2D.PREC)) |
                    (self.end0.is_subset_of(pt) & self.tmax <= Shape2D.PREC))

        # Is subset of, Arc to Line
        if isinstance(type(arg), Line):
            line = arg

            # Only possible if the arc radius = 0 or the arc length = 0
            return ((self.pt0.is_subset_of(line) & self.r <= Shape2D.PREC)) |
                    (self.end0.is_subset_of(line) & self.tmax <= Shape2D.PREC))

        # Is subset of, Arc to Ellipse
        if isinstance(type(arg), Ellipse):
            ellipse = arg

            # Require three points on arc to overlap Ellipse and to be unmasked
            self_pt2 = self.point_at(self.t0 + 0.5 * self.dt)
            ellipse0 = ellipse.point_at(ellipse.param_at(self.end0))
            ellipse1 = ellipse.point_at(ellipse.param_at(self.end1))
            ellipse2 = ellipse.point_at(ellipse.param_at(self_pt2))

            result = ((self.end0 - ellipse0).norm_sq() <= Shape2D.PRECx2) & \
                     ((self.end1 - ellipse1).norm_sq() <= Shape2D.PRECx2) & \
                     ((self_xpt  - ellipse2).norm_sq() <= Shape2D.PRECx2)

            result[result.mask & self.antimask & pt.antimask] = False

            return result

        # For other cases, use the method of the other object's class
        return arg.is_superset_of(self)

    def is_superset_of(self, arg):
        """True if this object is as superset of (i.e., entirely contains) the
        given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a superset of the given
                        shape.
        """

        # Is superset of, Arc to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            arcpt = self.point_at(self.param_at(pt))
            result = (arcpt - pt).norm_sq() <= Shape2D.PRECx2
            result[result.mask & self.antimask & pt.antimask] = False

            return result

        # Is superset of, Arc to Line or HalfLine
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        # Is superset of, Arc to Segment
        if isinstance(type(arg), Segment):
            line = arg

            # Only possible if the segment length = 0
            return ((line.pt0 - line.pt1).norm_sq() <= Shape2D.PREC)) &
                    (self.is_superset_of(line.pt0))

        # Is superset of, Arc to Ellipse
        if isinstance(type(arg), Ellipse):
            ellipse = arc
            return ellipse.is_subset_of(self)

        # For other cases, use the method of the other object's class
        return arg.is_superset_of(self)

    def touches(self, arg):
        """True if the this object and the given Shape2D touch but do not share
        any common interior points.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if the shapes touch but share no common
                        interior points.
        """

        # Touches, Arc to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            return arc.is_superset_of(pt)

        # Touches, Arc to Line (handled by Ellipse.touches())
        if isinstance(type(arg), Line):
            return super(Arc,self).touches(arg)

        # Otherwise use the general method
        return super(Shape2D, self).touches(arg)

################################################################################
