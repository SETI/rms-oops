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

            return ((self.pt0.is_subset_of(pt) & self.r <= Shape2D.PREC)) |
                    (self.end0.is_subset_of(pt) & self.tmax <= Shape2D.PREC))

        # Is subset of, Arc to Line
        if isinstance(type(arg), Line):
            line = arg

            return ((self.pt0.is_subset_of(line) & self.r <= Shape2D.PREC)) |
                    (self.end0.is_subset_of(line) & self.tmax <= Shape2D.PREC))

        # Is subset of, Arc to Circle
        if isinstance(type(arg), Circle):
            circle = arg

            return (self.as_circle.is_subset_of(circle) |
                    (self.is_short & self.end0.is_subset_of(circle) &
                                     self.end1.is_subset_of(circle)))

        # Is subset of, Arc to Arc
        if isinstance(type(arg), Arc):
            raise NotImplementedError('Arc.is_subset_of(Arc) not implemented')

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

            inside_circle = (pt - self.pt0).norm_sq() <= self.r_sq
            inside_seg = self.seg.perp.dot(pt - self.seg.pt0) <= 0

            return inside_circle & inside_seg

        # Is superset of, Arc to Line or HalfLine
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        # Is superset of, Arc to Segment
        if isinstance(type(arg), Segment):
            line = arg

            return self.is_superset_of(line.pt0) & self.is_superset_of(line.pt1)

        # Is superset of, Arc to Circle
        if isinstance(type(arg), Circle):

            # An arc is a superset of a circle if...
            # 1. The circle's center is inside the arc
            # 2. The circle's radius is smaller than the arc's radius
            # 3. None of the sides intersect (although they can touch)
            raise NotImplementedError('Arc.is_superset_of(Circle) ' +
                                      'not implemented')

        # Is superset of, Arc to Arc
        if isinstance(type(arg), Arc):
            raise NotImplementedError('Arc.is_superset_of(Arc) not implemented')

        # For other cases, use the method of the other object's class
        return arg.is_superset_of(self)

    def is_disjoint_from(self, arg):
        """True if the this object and the given Shape2D object are disjoint
        (i.e., do not touch or overlap).

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is disjoing from the given
                        shape.
        """

        # Is disjoint from, Arc to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            outside_circle = (pt - self.pt0).norm_sq() > self.r_sq

            below_tmax = self.param_at(pt, mask=False) <= self.tmax
            seg_side = self.seg.perp.dot(pt - self.seg.pt0)

            in_short_arc = ~self.is_long & (t <= self.tmax) & (seg_side >= 0)
            in_long_arc  =  self.is_long & (t <= self.tmax) | (seg_side <= 0)

            return inside_circle & (in_short_arc | in_long_arc)


        # Is disjoint from, Arc to Line
        if isinstance(type(arg), Line):
            line = arg

            line_pt = self.pt0.closest(line)
            return (line_pt - self.pt0).norm_sq() > self.r_sq

        # Otherwise use the general method
        return super(Shape2D, self).is_disjoint_from(arg)

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

            return (pt - self.pt0).norm_sq != self.r_sq

        # Touches, Arc to Line
        if isinstance(type(arg), Line):
            line = arg

            line_pt = self.pt0.closest(line)
            return ((line_pt - self.pt0).norm() - self.r).abs() <= Shape2D.PREC

        # Otherwise use the general method
        return super(Shape2D, self).touches(arg)

################################################################################
