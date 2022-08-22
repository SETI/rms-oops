################################################################################
# shape2d/line.py
################################################################################

import numpy as np
from polymath import *
from shape2d import Shape2D
from conic   import Conic
from point   import Point

#*******************************************************************************
# Line
#*******************************************************************************
class Line(Shape2D, Conic):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    An infinite line passing through a pair of points.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(pt0, pt1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Line object, which has no limits.

        Input:
            pt0         one point on the line, represented as a Pair of
                        arbitrary shape.
            pt1         another point on the line, represented as a Pair of
                        arbitrary shape.

        Note: The line is parameterized such that 0 corresponds to the first
        point and 1 corresponds to the second. The array shape of the Line
        object is the result of broadcasting together the array shapes of the
        two inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.pt0 = Pair.as_pair(pt0)
        self.pt1 = Pair.as_pair(pt1)

        self.dpt = self.pt1 - self.pt0
        self.r_sq = self.dpt.norm_sq()
        self.r    = self.r_sq.sqrt()
        self.r_sq_inv = 1. / self.r_sq
        self.r_inv    = 1. / self.r

        self.shape = self.dpt.shape

        self.tmin = -np.inf
        self.tmax =  np.inf

        self.perp = self.dpt.rot90(recursive=True)

        #--------------------------------
        # Fill in conic coefficients
        # d x + e y + f = 0
        # d dx + e dy = 0
        #--------------------------------
        (x,y) = self.pt0.to_scalars()
        (dx,dy) = self.dpt.to_scalars()

        self.a = Scalar.ZERO
        self.b = Scalar.ZERO
        self.c = Scalar.ZERO
        self.d =  dy
        self.e = -dx
        self.f = -(self.d * x + self.e * y)

        self.fill_conic_attributes()
    #===========================================================================



    #===========================================================================
    # from_conics
    #===========================================================================
    def from_conics(a,b,c,d,e,f):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Construct a Line object from the coefficients of a conic section:
                a x^2 + b xy + c y^2 + d x + e y + f = 0
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        a = Scalar.as_scalar(a)
        b = Scalar.as_scalar(b)
        c = Scalar.as_scalar(c)
        d = Scalar.as_scalar(d)
        e = Scalar.as_scalar(e)
        f = Scalar.as_scalar(f)

        if not ((a == 0).all() & (b == 0).all() & (c == 0).all()):
            raise ValueError('conic coefficients do not describe a Line')

        dx = -e
        dy =  d
        dpt = Pair.from_scalars(dx,dy)

        #---------------------------------------------------------
        # Solve for any (x,y) that satisfies d x + e y + f = 0
        #---------------------------------------------------------
        x0 = 0.
        y0 = -f/e

        y1 = 0.
        x1 = -f/d

        #-------------------------------------
        # Choose the more accurate option
        #-------------------------------------
        pt0 = Pair.from_scalars(x0,y0)
        pt1 = Pair.from_scalars(x1,y1)

        mask = d.abs() > e.abs()
        pt0[mask] = pt1[mask]

        return Line(pt0, pt0 + dpt)
    #===========================================================================



    ############################################################################
    # Tools to convert quickly between different subclasses
    ############################################################################

    #===========================================================================
    # clone
    #===========================================================================
    def clone(self, subtype=None):

        if subtype is None:
            subtype = type(self)

        obj = Line.__new__(subtype)
        for (key, value) in self.__dict__iteritems():
            obj.__dict__[key] = value
    #===========================================================================



    #===========================================================================
    # as_line
    #===========================================================================
    def as_line(self):
        obj = self.clone(subtype=Line)
        obj.tmin = -np.inf
        obj.tmax =  np.inf

        return obj
    #===========================================================================



    ############################################################################
    # Additional methods
    ############################################################################

    #===========================================================================
    # rotate
    #===========================================================================
    def rotate(self, angle):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Rotate a line from the center point through a specified angle
        counterclockwise.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        angle = Scalar.as_scalar(angle)
        return Line(self.pt0,
                    self.pt0 + self.dpt * angle.cos() + self.perp * angle.sin())
    #===========================================================================



    #===========================================================================
    # rotate90
    #===========================================================================
    def rotate90(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Rotate a line from the center point through 90 degrees
        counterclockwise. This is quicker than the general formula.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Line(self.pt0, self.pt0 + self.perp)
    #===========================================================================



    ############################################################################
    # Methods overridden by each Line subclass
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
    # Support methods
    ############################################################################

    #===========================================================================
    # _line_param_closest_to_point
    #===========================================================================
    def _line_param_closest_to_point(self, pt):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the parameter where the point on this line is closest to the
        given point. The returned parameter is clipped to the allowed range and
        left unmasked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-----------------------------------------------
        # Formula for distance line to point:
        #   line(t) = pt0 + t * dpt
        #
        # Closest point is perpendicular, where:
        #   dot(dpt, line(t) - pt) = 0
        #
        # Solve for t:
        #   dot(dpt, (pt0 - pt) + t * dpt) = 0
        #   dot(dpt, pt0 - pt) + t * dot(dpt, dpt) = 0
        #   t = dot(dpt, pt - pt0) / dot(dpt,dpt)
        #-----------------------------------------------
        t = self.dpt.dot(pt - self.pt0) * self.r_sq_inv
        return self._clip(t)
    #===========================================================================



    #===========================================================================
    # _line_intersection_params
    #===========================================================================
    def _line_intersection_params(self, line):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the parameter values where two lines intersect. Parameters
        are masked if the Lines or Line subclasses do not intersect.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------------------------
        # Search for an intersection:
        #   line0(s) = line0.pt0 + s * line0.dpt
        #   line1(t) = line1.pt0 + t * line1.dpt
        #
        #   line0.pt0 + s * line0.dpt = line1.pt0 + t * line1.dpt
        #
        # Solve for s and t:
        #  cross(line0.pt0, line1.dpt) + s * cross(line0.dpt, line1.dpt) =
        #                                    cross(line1.pt0, line1.dpt)
        #  s = (cross(line0.pt0, line1.dpt) - cross(line1.pt0, line1.dpt)) /
        #                                     cross(line0.dpt, line1.dpt)
        #  s = cross(line0.pt0 - line1.pt0, line1.dpt) /
        #                                     cross(line0.dpt, line1.dpt)
        #
        #  cross(line0.pt0, line0.dpt) = cross(line1.pt0, line0.dpt) +
        #                            t * cross(line1.dpt, line0.dpt)
        #  t = (cross(line1.pt0, line0.dpt) - cross(line0.pt0, line0.dpt)) /
        #                                     cross(line0.dpt, line1.dpt)
        #  t = cross(line1.pt0 - line0.pt0, line0.dpt) /
        #                                     cross(line0.dpt, line1.dpt)
        #
        # The result is masked if denom == 0, which implies parallel lines
        #--------------------------------------------------------------------
        denom_inv = 1. / self.pt0.cross(line.dpt)
        diff_pt0 = self.pt0 - line.pt0
        s = diff_pt0.cross(line.dpt) * denom_inv
        t = diff_pt0.cross(self.dpt) * denom_inv

        return (self._mask(s), line._mask(t))
    #===========================================================================



    #===========================================================================
    # _more_accurate_intersection
    #===========================================================================
    def _more_accurate_intersection(self, s, line, t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return the intersection points requiring shorter extrapolations.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        result0 = self.point_at(s)
        result1 = line.point_at(t)
        lmask = np.abs(t.values - 0.5) < np.abs(s.values - 0.5)

        result0[lmask] = result1[lmask]
        return Point(result0.reshape((1,) + result0.shape))
    #===========================================================================



    #===========================================================================
    # _line_conic_intersection_params
    #===========================================================================
    def _line_conic_intersection_params(self, conic, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Find the intersection points between this line and a conic.

        Inputs:
            conic       a Conic represented by N object with six Scalar
                        attributes (a,b,c,d,e,f)
            recursive   True to evaluate derivatives at the intersections.

        Return:         A tuple of two Scalars containing the parameter values
                        at which the Line intersects an arbitrary Conic. For
                        lines that do not intersect the Conic, both Scalar
                        values are masked. For those that only intersect the
                        conic once, the second value is masked. When both
                        intersections exist, the greater value of the parameter
                        comes second.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #--------------------------------------------------------------------
        # This Line: x = X0 + t * DX
        #            y = Y0 + t * DY
        #
        # Conic described by:
        #   a x2 + b xy + c y2 + d x + e y + f = 0
        #
        # Solve for t. This is a quadratic equation.
        #
        # 0 = ( a DX DX +  b DX DY +             c DY DY) t2 +
        #     (2a X0 DX + 2b DX Y0 + 2b X0 DY + 2c Y0 DY + d DX + e DY) t +
        #     ( a X0 X0 +  b X0 Y0 +             c Y0 Y0 + d X0 + e Y0 + f
        #--------------------------------------------------------------------
        (X0, Y0) = self.pt0.to_scalars(recursive)
        (DX, DY) = self.dpt.to_scalars(recursive)

        (a,b,c,d,e,f) = conic.abcdef[recursive]

        A =     a*DX**2 + b*DX*DY           + c*DY**2
        B = 2.*(a*X0*DX + b*(DX*Y0 + X0*DY) + c*Y0*DY) + d*DX + e*DY
        C =     a*X0*X0 + b*X0*Y0           + c*Y0**2  + d*X0 + e*Y0 + f

        roots = Polynomial.from_scalars(A,B,C).roots()
        return (roots[0], roots[1])
    #===========================================================================



    ############################################################################
    # Standard methods
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
        return Scalar.ONE
    #===========================================================================



    #===========================================================================
    # is_convex
    #===========================================================================
    def is_convex(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the shape is convex.
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
        Parameterization of the shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return Point(self.pt0 + self._mask(t) * self.dpt)
    #===========================================================================



    #===========================================================================
    # param_at
    #===========================================================================
    def param_at(pt):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Parameter at a point, which is assumed to fall on the edge of this
        object.

        What happens when the point does not fall on the shape is undetermined.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self._line_param_closest_to_point(pt)
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
        are guaranteed to be unmasked as long as the shape are initially
        unmasked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-----------------------------
        # Closest, Line to Point
        #-----------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            t = self._line_param_closest_to_point(pt)
            return (self.point_at(t), pt)

        #---------------------------
        # Closest, Line to Line
        #---------------------------
        if isinstance(type(arg), Line):
            line = arg

            #- - - - - - - - - - - - - - - - - - - - - - -
            # Locate intersection; masked if necessary
            #- - - - - - - - - - - - - - - - - - - - - - -
            (s,t) = self._line_intersection_params(line)
            self_pts = [self._more_accurate_intersection(s, line, t)]
            line_pts = [self_pts[0]]

            #- - - - - - - - - - - - - - - - - - - - - - - - - -
            # Return the easy solution now for infinite lines
            #- - - - - - - - - - - - - - - - - - - - - - - - - -
            if type(self) == Line and type(arg) == Line:
                return (self_pts[0], line_pts[0])

            #- - - - - - - - - - - - - - -
            # Also consider endpoints
            #- - - - - - - - - - - - - - -
            if type(self) in (HalfLine, Segment):
                (line_pt, self_pt) = line.closest(self.pt0)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(self) == Segment:
                (line_pt, self_pt) = line.closest(self.pt1)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(line) in (HalfLine, Segment):
                (self_pt, line_pt) = self.closest(line.pt0)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            if type(line) == Segment:
                (self_pt, line_pt) = self.closest(line.pt1)
                self_pts.append(self_pt)
                line_pts.append(line_pt)

            #- - - - - - - - - - - - - - - - - - - - -
            # Select and return closest pairings
            #- - - - - - - - - - - - - - - - - - - - -
            return Shape2D._closest_of_pairings(self_pts, line_pts)

        #--------------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #--------------------------------------------------------------------
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

        #----------------------------------
        # Intersections, Line to Point
        #----------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            (pt0, pt1) = self.closest(pt)
            pairs = pt0.mask_where(pt1 != pt0)
            return Point(pairs.reshape((1,) + pairs.shape))

        #-------------------------------
        # Intersections, Line to Line
        #-------------------------------
        if isinstance(type(arg), Line):
            line = arg

            (s,t) = self._line_intersection_params(line)
            result = self._more_accurate_intersection(s, line, t)
            return Point(result.reshape((1,) + result.shape))

        #--------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #--------------------------------------------------------------
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
        all_masked = pt.as_all_masked()
        return (all_masked, all_masked)
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
        return Line(self.pt0, self.pt1)
    #===========================================================================



    #===========================================================================
    # normal_at
    #===========================================================================
    def normal_at(self, t):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The outward HalfLine object normal to this Shape2D object at the
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pt0 = self.point_at(t)
        pt1 = pt0 + self.perp
        return HalfLine(pt0, pt1)
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

        #-------------------------------
        # Is subset of, Line to Point
        #-------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            return (self.pt0 == pt) & (self.pt1 == pt) & (type(self) == Segment)

        #--------------------------------
        # Is subset of, Line to Line
        #--------------------------------
        if isinstance(type(arg), Line):
            line = arg

            (line_pt0, self_pt0) = line.closest(self.pt0)
            (line_pt1, self_pt1) = line.closest(self.pt1)
            return (line_pt0 == self_pt0) & (line_pt1 == self_pt1)

        #----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #----------------------------------------------------------------
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

        #------------------------------------
        # Is superset of, Line to Point
        #------------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            (line_pt, point_pt) = line.closest(pt)
            return (line_pt == point_pt)

        #-----------------------------------
        # Is superset of, Line to Line
        #-----------------------------------
        if isinstance(type(arg), Line):
            line = arg

            (self_pt0, line_pt0) = self.closest(line.pt0)
            (self_pt1, line_pt1) = self.closest(line.pt1)
            return (self_pt0 == line_pt0) & (self_pt1 == line_pt1)

        #----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #----------------------------------------------------------------
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

        #------------------------------------
        # Is disjoint from, Line to Point
        #------------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            (line_pt, point_pt) = line.closest(pt)
            return (line_pt != point_pt)

        #------------------------------------
        # Is disjoint from, Line to Line
        #------------------------------------
        if isinstance(type(arg), Line):
            line = arg

            pts = self.intersections(line)[0]
            return Boolean(np.logical_not(pts.mask))

        #----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #----------------------------------------------------------------
        return arg.is_disjoint_from(self)
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
        # Touches, Line to Point
        #------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)
            (line_pt, point_pt) = line.closest(pt)
            return (line_pt - point_pt).norm_sq() <= Shape2D.PREC_SQ

        #---------------------------
        # Touches, Line to Line
        #---------------------------
        if isinstance(type(arg), Line):
            line = arg

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # True if exactly one endpoint falls on the other line
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            touchings = []

            if type(self) in (HalfLine, Segment):
              (line_pt, self_pt) = line.closest(self.pt0)
              touchings.append((line_pt - self_pt).norm_sq() <= Shape2D.PREC_SQ)

            if type(self) == Segment:
              (line_pt, self_pt) = line.closest(self.pt1)
              touchings.append((line_pt - self_pt).norm_sq() <= Shape2D.PREC_SQ)

            if type(line) in (HalfLine, Segment):
              (self_pt, line_pt) = self.closest(line.pt0)
              touchings.append((line_pt - self_pt).norm_sq() <= Shape2D.PREC_SQ)

            if type(line) == Segment:
              (self_pt, line_pt) = self.closest(line.pt1)
              touchings.append((line_pt - self_pt).norm_sq() <= Shape2D.PREC_SQ)

            touchings = Qube.stack(*touchings)
            return (touchings.sum(axis=0) == 1)

        #----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #----------------------------------------------------------------
        return arg.touches(self)
    #===========================================================================


#*******************************************************************************



################################################################################
################################################################################
# Subclass HalfLine
################################################################################
################################################################################

#*******************************************************************************
# HalfLine
#*******************************************************************************
class HalfLine(Line):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    An semi-infinite line starting from one point and passing through
    another.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(pt0, pt1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a HalfLine object, which has a limit in only one
        direction.

        Input:
            pt0         one point on the line, represented as a Pair of
                        arbitrary shape. This is the origin.
            pt1         another point on the line, represented as a Pair of
                        arbitrary shape.

        Note: The line is parameterized such that 0 corresponds to the first
        point and 1 corresponds to the second. The line only exists for t >= 0.
        The array shape of the HalfLine object is the result of broadcasting
        together the array shapes of the two inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        super(Line, self).__init__()
        self.tmin = 0.
    #===========================================================================



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
            return obj.mask_where(t < 0)

        return Scalar.as_scalar(t).mask_where_lt(0.)
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
        return Scalar.as_scalar(t).clip(0., None, remask=False)
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
        return (self.pt0, self.pt0.as_all_masked())
    #===========================================================================


#*******************************************************************************


################################################################################
################################################################################
# Subclass Segment
################################################################################
################################################################################

#*******************************************************************************
# Segment
#*******************************************************************************
class Segment(Line):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    An line segment from one point to another.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(pt0, pt1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Segment object, which is a straight line between
        two endpoints.

        Input:
            pt0         one point on the line, represented as a Pair of
                        arbitrary shape. This is the origin.
            pt1         another point on the line, represented as a Pair of
                        arbitrary shape. This is the endpoint

        Note: The line is parameterized such that 0 corresponds to the first
        point and 1 corresponds to the second. The line only exists for t
        between 0 and 1. The array shape of the Segment object is the result of
        broadcasting together the array shapes of the two inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        super(Line, self).__init__()
        self.tmin = 0.
        self.tmax = 1.
    #===========================================================================



    #===========================================================================
    # _mask
    #===========================================================================
    def _mask(self, t, obj=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Masks where Scalar t is outside the allowed range.

        Input:
            t       parameter of shape.
            obj     the object to mask and return; if None, then t is masked and
                    returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if obj is not None:
            return obj.mask_where((t < 0) | (t > 1))

        return Scalar.as_scalar(t).mask_where_outside(0., 1.)
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
        return Scalar.as_scalar(t).clip(0., 1., remask=False)
    #===========================================================================



    #===========================================================================
    # dimensions
    #===========================================================================
    # Needs to override the default method because a line where the endpoints
    # match as dimension zero.
    def dimensions(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The Scalar dimension of this object: 0 for a point; 1 for a line; 2
        for a shape object that has nonzero area.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if (self.r == 0).any():
            return (self.r != 0).as_int()

        return Scalar.ONE
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
        return (self.pt0, self.pt1)
    #===========================================================================


#*******************************************************************************


################################################################################
# Once the load is complete, we can fill in a reference to the Line and
# HalfLine classes inside the Point object.
################################################################################

Point.LINE_CLASS = Line
Point.HALFLINE_CLASS = HalfLine

Line.HALFLINE_CLASS = HalfLine
Line.SEGMENT_CLASS = Segment

################################################################################
