################################################################################
# shape2d/triangle.py
################################################################################

import numpy as np
from polymath import *
from shape2d import Shape2D
from point   import Point
from line    import Line, HalfLine, Segment
from circle  import Circle

class Triangle(Shape2D):
    """A triangle."""

    #===========================================================================
    def __init__(pt0, pt1, pt2):
        """Constructor for a Triangle object, defined by three corners.

        Input:
            pt0         one corner of the triangle, represented as a Pair of
                        arbitrary shape.
            pt1         second corner.
            pt2         third corner.

        Note: The triangle is parameterized such that 0 corresponds to the first
        point, 1 corresponds to the second, and 2 corresponds to the third.
        The object's array shape is the result of broadcasting together the
        array shapes of the three inputs.
        """

        self.pt0 = Pair.as_pair(pt0)
        self.pt1 = Pair.as_pair(pt1)
        self.pt2 = Pair.as_pair(pt2)
        self.pts = Qube.stack(self.pt0, self.pt1, self.pt2, self.pt0)
        self.pts3 = self.pts[:3]

        self.dpt = self.pts[1:] - self.pts3

        self.seg0 = Segment(self.pt0, self.pt1)
        self.seg1 = Segment(self.pt1, self.pt2)
        self.seg2 = Segment(self.pt2, self.pt0)
        self.segs = [self.seg0, self.seg1, self.seg2, self.seg0]

        self.sign = self.seg0.dpt.cross(self.seg1.dpt).sign()
        # A positive sign indicates a triangle drawn in the counterclockwise
        # direction; a negative sign indicates a triangle drawn in the clockwise
        # direction. Zero indicates a zero-area triangle.

        # Used by is_superset_of() for circles
        self.half_longest = None

    ############################################################################
    # Standard methods
    ############################################################################

    def dimensions(self):
        """The Scalar dimension of this object: 0 for a point; 1 for a line; 2
        for a shape object that has nonzero area.
        """

        if (self.r <= Shape2D.PREC).any():
            return (self.r > Shape2D.PREC).as_int()

        return Scalar.TWO

    #===========================================================================
    def is_convex(self):
        """True if the shape is convex."""

        return Boolean.TRUE

    #===========================================================================
    def point_at(t):
        """Parameterization of the shape."""

        t = Scalar.as_scalar(t) % 3.
        k = t.as_int()
        t = t - k

        return Point(self.pts[k] + t * self.dpt[k])

    #===========================================================================
    def param_at(pt):
        """Parameter at a point, which is assumed to fall on the edge of this
        object.

        What happens when the point does not fall on the shape is undetermined.
        """

        edge_pts = Qube.stack(self.seg0.closest(pt)[0],
                              self.seg1.closest(pt)[0],
                              self.seg2.closest(pt)[0])
        norm_sq = (edge_pts - pt).norm_sq()
        argmin = norm_sq.argmin(axis=0)
        indx = Shape2D._meshgrid_for_arg(argmin)

        params = Qube.stack(self.seg0.param_at(pt), self.seg1.param_at(pt),
                                                    self.seg2.param_at(pt))
        return params[indx]

    #===========================================================================
    def param_limits(self):
        """Parameter limits to define the shape."""

        return (0., 3.)

    #===========================================================================
    def closest(self, arg):
        """Tuple containing the pairs of closest points between the edges of
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
        are guaranteed to be unmasked as long as the shape are initially
        unmasked.
        """

        (self_pt0, arg_pt0) = self.seg0.closest(arg)
        (self_pt1, arg_pt1) = self.seg1.closest(arg)
        (self_pt2, arg_pt2) = self.seg2.closets(arg)

        self_pts = Qube.stack(self_pt0, self_pt1, self_pt2)
        arg_pts  = Qube.stack(arg_pt0,  arg_pt1,  arg_pt2)

        return Shape2D._closest_of_pairings(self_pts, arg_pts)

    #===========================================================================
    def intersections(self, arg):
        """Points defining intersections between the edges of this shape and the
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

        # Find intersections on all three segments
        xsect0 = self.seg0.intersect(arg)
        xsect1 = self.seg1.intersect(arg)
        xsect2 = self.seg2.intersect(arg)

        # Stack and reshape so all intersections fall along axis 0
        xsects = Qube.stack(xsect0, xsect1, xsect2)
        axis0 = xsects.shape[0] * xsects.shape[1]
        new_shape = (axis0,) + xsects.shape[2:]

        # Create a new mask to hide duplicated values
        new_mask = xsects.mask.copy()
        x = xsects.values[...,0]
        y = xsects.values[...,1]
        unmasked = xsects.antimask

        # Work from right...
        for k in range(axis0-1,-1,0):

          # Compare each value to all values to its left on axis 0
          for j in range(k-1,-1,-1):

            # If the left value (j) is equal to the right value (k) and the
            # left value is unmasked, mask the right value
            new_mask[k] |= ((x[j] - x[k]).abs() <= Shape2D.PREC &
                            (y[j] - y[k]).abs() <= Shape2D.PREC &
                            unmasked[j])

        return xsects.mask_where(new_mask)

    #===========================================================================
    def tangents_from(self, pt):
        """The two points where this Shape2D object is tangent to a line from
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

        # Select the two extremes among the three corners
        corner0 = (self.pt0 - pt).unit()
        corner1 = (self.pt1 - pt).unit()
        corner2 = (self.pt2 - pt).unit()

        # The two most widely separated vectors will have the cross product
        # with the largest absolute value
        sep10 = corner1.cross(corner0).abs()
        sep20 = corner2.cross(corner0).abs()
        sep21 = corner2.cross(corner1).abs()

        seps = Qube.stack(sep10, sep20, sep21)
        argmax = seps.argmax(axis=0)
        indx = Shape2D._meshgrid_for_arg(argmax)

        first  = Qube.stack(self.pt1, self.pt2, self.pt2)
        second = Qube.stack(self.pt0, self.pt0, self.pt1)
        return (first[indx], second[indx])

    #===========================================================================
    def tangent_at(self, t):
        """The Line object tangent to this Shape2D object at the given parameter
        value.

        Input:
            self        this shape.
            t           a parameter value on this shape.

        Return:         a Line object defining tangent points. The array shapes
                        of self and t are broadcasted together and the returned
                        result is a Line object with this shape. Tangent lines
                        will be masked if the tangent value is undefined.
        """

        pts = self.point_at(t)
        side = (t % 3).as_int()
        pt0 = self.pts[side]
        pt1 = self.pts[side + 1]

        pt0 = pt0.mask_where((side - t).abs() < Shape2D.PREC)
                                            # corner points have no tangent

        return Line(pt0, pt1)

    #===========================================================================
    def normal_at(self, t):
        """The outward HalfLine object normal to this Shape2D object at the
        given parameter value.

        Note: for Line subclasses, the "outward" normal is defined to be the
        normal facing leftward as one transits the line in the direction in
        which the parameter increases.

        Input:
            self        this shape.
            t           a parameter value on this shape.

        Return:         a HalfLine object defining the outward normal. The array
                        shapes of self and t are broadcasted together and the
                        returned result is a HalfLine object with this shape.
                        The HalfLines will be masked if the outward normal is
                        undefined.
        """

        pt0 = self.point_at(t)
        side = (t % 3).as_int()
        dpt = -self.dpt[side] * self.sign
           # rotate90 will rotate counterclockwise and so point inward for any
           # triangle defined in a counterclockwise direction. This ensures that
           # all lines will point outward after rotate90.

        pt0 = pt0.mask_where((side - t).abs() < Shape2D.PREC)
                                            # corner points have no normal

        return HalfLine(pt0, pt0 + dpt).rotate90()

    #===========================================================================
    def is_subset_of(self, arg):
        """True if this object is as subset of (i.e., is entirely contained by)
        the given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a subset of the given
                        shape.
        """

        # Convex shapes are easy
        if type(arg) == Pair:
            pt = Point(arg)

        if arg.is_convex().all():
            return (self.pt0.is_subset_of(arg)) & \
                   (self.pt1.is_subset_of(arg)) & \
                   (self.pt2.is_subset_of(arg))

        # For other cases, use the method of the other object's class
        return arg.is_superset_of(self)

    #===========================================================================
    def is_superset_of(self, arg):
        """True if this object is as superset of (i.e., entirely contains) the
        given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a superset of the given
                        shape.
        """

        # Is superset of, Triangle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            crosses = self.dpts.cross(pt - self.pts3)
            return (crosses * self.signs >= 0).all(axis=0)

        # Is superset of, Triangle to Line
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        if type(arg) == Segment:
            line = arg
            return self.is_superset_of(line.pt0) & self.is_superset_of(line.pt1)

        # Is superset of, Triangle to Circle
        if type(arg) == Circle:
            circle = arg

            # A triangle is a superset of a circle if...
            # 1. The circle's center is inside the triangle
            # 2. The circle's diameter is smaller than the triangle's longest
            #    side.
            # 3. None of the three sides intersects the circle (although they
            #    can touch)

            circle_center_is_inside = self.is_superset_of(circle.pt0)

            if self.half_longest is None:
                lengths = Qube.stack(self.seg0.r, self.seg1.r, self.seg2.r)
                self.half_longest = lengths.max(axis=0) / 2.

            circle_is_small_enough = self.half_longest < circle.r

            # Second index [1] is masked when circle and segment just touch
            crosses_side01 = self.seg01.intersections(circle)[1].antimask
            crosses_side12 = self.seg12.intersections(circle)[1].antimask
            crosses_side20 = self.seg20.intersections(circle)[1].antimask
            circle_doesnt_cross_side = np.logical_not(crosses_side01 |
                                                      crosses_side12 |
                                                      crosses_side20)

            return (circle_center_is_inside & circle_is_small_enough &
                    circle_doesnt_cross_side)

        # Is superset of, Triangle to Triangle
        if type(arg) == Triangle:
            triangle = arg

            return triangle.is_subset_of(self)

        # For other cases, use the method of the other object's class
        return arg.is_superset_of(self)

    #===========================================================================
    def is_disjoint_from(self, arg):
        """True if the this object and the given Shape2D object are disjoint
        (i.e., do not touch or overlap).

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is disjoing from the given
                        shape.
        """

        # Is disjoint from, Triangle to Point
        if type(arg) in (Point, Pair):
            return ~self.is_superset_of(arg)

        # Is disjoint from, Triangle to Line
        if type(arg) == Line:
            line = arg

            # A triangle is disjoint from an infinite line if all three points
            # fall on the same side of the line.

            diffs = self.pts3 - line.pt0
            sides = diffs.dot(line.perp)
            return (sides >= 0.).all(axis=0) | (sides <= 0.).all(axis=0)

        if type(arg) in (HalfLine, Segment):
            line == arg

            # A triangle is disjoint from a half-line or segment if the full
            # line would be disjoint or if one endpoint is outside the triangle

            return (self.is_disjoint_from(Line.as_line(line)) |
                    self.is_disjoin_from(line.pt0))

        # Otherwise use the general method
        return super(Shape2D, self).is_disjoint_from(arg)

    #===========================================================================
    def touches(self, arg):
        """True if the this object and the given Shape2D touch but do not share
        any common interior points.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if the shapes touch but share no common
                        interior points.
        """

        # Touches, Triangle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            return (self.seg01.touches(pt) | self.seg12.touches(pt) |
                                             self.seg23.touches(pt))

        # Touches, Triangle to Line
        if isinstance(type(arg), Line):
            line = arg

            # True if exactly one line endpoint touches the triangle or if the
            # line touches one corner
            touchings = [line.touches(self.pt0), line.touches(self.pt1),
                                                 line.touches(self.pt2)]

            # Count matches on the side but not the corners
            (side_pt, line_pt) = self.seg01.closest(line)
            touches_side = (side_pt - line_pt).norm_sq() <= Shape2D.PREC_SQ
            mask = ((side_pt - self.pt0).norm_sq() <= Shape2D.PREC_SQ) |
                    (side_pt - self.pt1).norm_sq() <= Shape2D.PREC_SQ))
            touches_side[mask] = False
            touchings.append(touches_side)

            (side_pt, line_pt) = self.seg12.closest(line)
            touches_side = (side_pt - line_pt).norm_sq() <= Shape2D.PREC_SQ
            mask = ((side_pt - self.pt1).norm_sq() <= Shape2D.PREC_SQ) |
                    (side_pt - self.pt2).norm_sq() <= Shape2D.PREC_SQ))
            touches_side[mask] = False
            touchings.append(touches_side)

            (side_pt, line_pt) = self.seg20.closest(line)
            touches_side = (side_pt - line_pt).norm_sq() <= Shape2D.PREC_SQ
            mask = ((side_pt - self.pt2).norm_sq() <= Shape2D.PREC_SQ) |
                    (side_pt - self.pt0).norm_sq() <= Shape2D.PREC_SQ))
            touches_side[mask] = False
            touchings.append(touches_side)

            touchings = Qube.stack(*touchings)
            return (touchings.sum(axis=0) == 1)

        # Otherwise use the general method
        return super(Shape2D, self).touches(arg)

################################################################################
