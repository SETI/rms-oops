################################################################################
# shapes2d/circle.py
################################################################################

import numpy as np
from polymath import *
from shape2d  import Shape2D
from conic    import Conic

from point    import Point
from line     import Line, HalfLine, Segment

class Circle(Shape2D, Ellipse):
    """A circle."""

    def from_conics(a,b,c,d,e,f):
        """Construct a Circle object from the coefficients of a conic section:
                a x^2 + b xy + c y^2 + d x + e y + f = 0
        """

        a = Scalar.as_scalar(a)
        b = Scalar.as_scalar(b)
        c = Scalar.as_scalar(c)
        d = Scalar.as_scalar(d)
        e = Scalar.as_scalar(e)
        f = Scalar.as_scalar(f)

        if not ((a == c).all() & (a != 0).all() & (b != 0).all()):
            raise ValueError('conic coefficients do not describe a Circle')

        # a (x - x0)^2 + a (y - y0)^2 - a r_sq = 0
        # a x^2 - 2a x0 x + a x0^2 + a y^2 - 2a y0 y + a y0^2 - a r_sq = 0
        # a x^2 + a y^2 + (-2a x0) x + (-2a y0) y + a (x0^2 + y0^2 - r_sq) = 0
        #
        # Solve for x0, y0, r_sq

        neg_half_a_inv = -0.5 / a
        x0 = neg_half_a_inv * d
        y0 = neg_half_a_inv * e
        r_sq = x0*x0 + y0*y0 + 2 * neg_half_a_inv * f

        if not (r_sq >= 0.).all():
            raise ValueError('conic coefficients do not describe a Circle')

        pt0 = Pair.from_scalars(x0, y0)
        pt1 = Pair.from_scalars(x0 + r_sq.sqrt(), y0)
        return Circle(pt0, pt1)

    ############################################################################
    # Methods overridden by each Line subclass
    ############################################################################

    def _mask(self, t, obj=None):
        """Masks where Scalar t is outside the allowed range.

        Input:
            t       parameter of shape.
            obj     the object to mask and return; if None, then t is masked and
                    returned.
        """

        if obj is not None:
            return obj

        return t

    #===========================================================================
    def _clip(self, t):
        """Clips Scalar t to the allowed range. No masking is applied."""

        return t

    ############################################################################
    # Methods defined for all classes
    ############################################################################

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

        # Closest, Circle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            # Draw a radial vector from the circle center through the point
            halfline = HalfLine(self.pt0, pt)
            self_pt = halfline.point_at(self.r * halfline.r_inv)

            if type(self) == Arc:
                self_pt = self._mask_pt(self_pt)
                self_pts = Qube.stack(self_pt, self.end0, self.end1)

                norm_sq = (self_pts - pt).norm_sq()
                argmin = norm_sq.argmin(axis=0)
                indx = Shape2D._meshgrid_for_arg(argmin)
                self_pt = self_pts[indx]

            return (self_pt, pt)

        # Closest, Circle to Line
        if isinstance(self, Line):
            line = arg

            # Find intersections
            (s0, s1) = Circle._line_intersection_params(self, line)
            self_pts = [line.point_at(s0), line.point_at(s1)]
            line_pts = list(self_pts)   # a copy

            # Find point on line closest to center
            s = Line._line_param_closest_to_point(line, self.pt0)
            line_pt = line.point_at(s)
            halfline = HalfLine(line_pt, self.pt0)
            self_pt = halfline.point_at(self.r * halfline.r_inv)
            self_pts.append(self_pt)
            line_pts.append(line_pt)

            # Also consider endpoints
            if type(line) in (HalfLine, Segment):
                (self_pt, line_pt) = self.closest(line.pt0)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(line) == Segment:
                (self_pt, line_pt) = self.closest(line.pt1)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(self) == Arc:
                (self_pt0, line_pt0) = self.end0.closest(line)
                (self_pt1, line_pt1) = self.end1.closest(line)
                self_pts += [self_pt0, self_pt1]
                line_pts += [line_pt0, line_pt1]

            # Select and return closest pairings
            return Shape2D._closest_of_pairings(self_pts, line_pts)

        # Closest, Circle to Circle
        if isinstance(self, Circle):
            circle = arg

            # Find intersections
            (t0, t1) = self._circle_intersection_params(circle)
            self_pt0 = self.point_at(t0)
            circle_pt0 = self_pt0

            # Find closest points on circles
            self_pt1 = self.point_at(self.param_at(circle.pt0))
            circle_pt1 = circle.point_at(circle.param_at(self.pt0))

            # Select and return closest pairings
            return Shape2D._closest_of_pairings([self_pt0, self_pt1],
                                                [circle_pt0, circle_pt1])

        # For other cases, use the method of the other object's class
        return arg.closest(self)[::-1]

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

        # Intersections, Circle to Point
        if type(arg) == Point:
            pt = arg

            norm_sq = (pt - self.pt0).norm_sq()
            points = Point(self.arg.mask_where(norm_sq != self.r_sq))
            points = pairs.mask_pts(points)
            return Point(points.reshape((1,) + pairs.shape))

        # Intersections, Circle to Line
        if isinstance(arg, Line):
            line = arg

            (s0, s1) = Circle._line_intersection_params(self, line)
            pt0 = arg.point_at(s0)
            pt1 = arg.point_at(s1)

            return Point(Qube.stack(pt0, pt1))

        # Intersections, Circle to Circle
        if isinstance(arg, Circle):
            circle = arg

            (t0, t1) = self._circle_intersection_params(circle)
            return Point(Qube.stack(self.point_at(t0), self.point_at(t1)))

        # For other cases, use the method of the other object's class
        return arg.intersections(self)

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

        # Determine the parameters of the tangent points
        dangle = (self.r / (pt - self.pt0).norm()).arccos()
        angle0 = self.param_at(pt)      # = param at a line from the center

        return (self.point_at(angle0 - dangle), self.point_at(angle0 + dangle))

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

        # Construct a line from this point to the center
        line = (self.point_at(t), self.pt0)

        # Rotate 90 degrees
        return line.rotate90()

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
        return HalfLine(pt0, pt0 + (pt0 - self.pt0))

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

        # Is subset of, Circle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            return (pt - self.pt0).norm() - self.r <= Shape2D.PREC)

        # Is subset of, Circle to Line
        if isinstance(type(arg), Line):
            line = arg

            return self.pt0.is_subset_of(line) & (self.r <= Shape2D.PREC)

        # Is subset of, Circle to Circle
        if isinstance(type(arg), Circle):
            circle = arg

            dist = (circle.pt0 - self.pt0).norm()
            return (dist + self.r <= circle.r)

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

        # Is superset of, Circle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            return (pt - self.pt0).norm_sq() <= self.r_sq

        # Is superset of, Circle to Line or HalfLine
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        # Is superset of, Circle to Segment
        if isinstance(type(arg), Segment):
            line = arg

            return self.is_superset_of(line.pt0) & self.is_superset_of(line.pt1)

        # Is superset of, Circle to Circle
        if isinstance(type(arg), Circle):
            circl = arg

            return circle.is_subset_of(self)

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

        # Is disjoint from, Circle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            return (pt - self.pt0).norm_sq() > self.r_sq

        # Is disjoint from, Circle to Line
        if isinstance(type(arg), Line):
            line = arg

            line_pt = self.pt0.closest(line)
            return (line_pt - self.pt0).norm_sq() > self.r_sq

        # Is disjoint from, Circle to Circle
        if isinstance(type(arg), Circle):
            circle = arg

            return (circle.pt0 - self.pt0).norm() > self.r + circle.r

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

        # Touches, Circle to Point
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            return ((pt - self.pt0).norm() - self.r).abs() <= Shape2D.PREC

        # Touches, Circle to Line
        if isinstance(type(arg), Line):
            line = arg

            raise NotImplemented('TBD')

        # Touches, Circle to Circle
        if isinstance(type(arg), Circle):
            circle = arg

            return ((circle.pt0 -
                     self.pt0).norm() - self.r - circle.r).abs() <= Shape2D.PREC

        # Otherwise use the general method
        return super(Shape2D, self).touches(arg)

################################################################################
