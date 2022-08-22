################################################################################
# shape2d/polygon.py
################################################################################

import numpy as np
from polymath import *
from shape2d  import Shape2D
from point    import Point
from line     import Line, HalfLine, Segment
from circle   import Circle
from triangle import Triangle

#*******************************************************************************
# Polygon
#*******************************************************************************
class Polygon(Shape2D):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A triangle.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(*pts):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for an abitrary polygon object, defined by three or more
        corners.

        Input:
            pts         a tuple of Point objects

        Note: The polygon is parameterized such that 0 corresponds to the first
        point, 1 corresponds to the second, etc.

        The object's array shape is the result of broadcasting together the
        array shapes of the inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        pt_list = [Pair.as_pair(pts[0]), Pair.as_pair(pts[1])]
        self.segs = [Segment(pt_list[0], pt_list[1])]

        self.triangles = []
        for pt in pts[2:]:
            pt = Pair.as_pair(pt)
            pt_list.append(pt)
            self.segs.append(Segment(pt_list[-2], pt_list[-1]))
            self.triangles = Triangle(pt_list[0], pt_list[1], pt)

        #----------------------
        # Close loop
        #----------------------
        pt_list.append(self.pts[0])
        self.pts = Qube.stack(*self.pts)
        self.ptsn = self.pts[:-1]

        self.dpt = self.pts[1:] - self.ptsn
        self.n = len(self.pts)
        self.nfloat = float(self.n)

        #-----------------------
        # Other properties
        #-----------------------
        dims = []
        signs = []
        for triangle in self.triangles:
            dims.append(Scalar.as_scalar(triangle.dimensions()))
            signs.append(Scalar.as_scalar(triangle.sign))

        dims = Qube.stack(dims)
        self.dims = dims.max(axis=0)

        signs = Qube.stack(signs)
        self.max_signs = signs.max(axis=0)
        self.min_signs = signs.min(axis=0)
        self.is_convex = (self.max_signs > 0) & (self.min_signs < 0)
        self.sign = signs.sum(axis=0).sign()

        #--------------------------------------------------
        # Concave polygons are not currently supported
        #--------------------------------------------------
        if not self.is_convex.all():
            raise ValueError('concave polygons are not supported')
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
        return self.dims
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
        return self.is_convex
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
        t = Scalar.as_scalar(t) % self.nfloat
        k = t.as_int()
        t = t - k

        return Point(self.pts[k] + t * self.dpt[k])
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
        edge_pts = []
        for seg in self.segs:
            edge_pts.append(seg.closest(pt)[0])
        edge_pts = Qube.stack(*edge_pts)

        norm_sq = (edge_pts - pt).norm_sq()
        argmin = norm_sq.argmin(axis=0)
        indx = Shape2D._meshgrid_for_arg(argmin)

        params = []
        for seg in self.segs:
            params.append(seg.param_at(pt))
        params = Qube.stack(*params)

        return params[indx]
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
        return (0., self.nfloat)
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
        self_pts = []
        arg_pts = []
        for seg in self.segs:
            (self_pt, arg_pt) = seg.closest(arg)
            self_pts.append(self_pt)
            arg_pts.append(arg_pt)

        self_pts = Qube.stack(*self_pts)
        arg_pts  = Qube.stack(*arg_pts)

        return Shape2D._closest_of_pairings(self_pts, arg_pts)
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

        #-----------------------------------------
        # Find intersections on all segments
        #-----------------------------------------
        xsects = []
        for seg in self.segs:
            xsects.append(seg.intersect(arg))

        #-------------------------------------------------------------
        # Stack and reshape so all intersections fall along axis 0
        #-------------------------------------------------------------
        xsects = Qube.stack(*xsects)
        axis0 = xsects.shape[0] * xsects.shape[1]
        new_shape = (axis0,) + xsects.shape[2:]

        #--------------------------------------------------
        # Create a new mask to hide duplicated values
        #--------------------------------------------------
        new_mask = xsects.mask.copy()
        x = xsects.values[...,0]
        y = xsects.values[...,1]
        unmasked = xsects.antimask

        #------------------------
        # Work from right...
        #------------------------
        for k in range(axis0-1,-1,0):

          #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          # Compare each value to all values to its left on axis 0
          #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          for j in range(k-1,-1,-1):

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # If the left value (j) is equal to the right value (k) and the
            # left value is unmasked, mask the right value
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask[k] |= ((x[j] - x[k]).abs() <= Shape2D.PREC &
                            (y[j] - y[k]).abs() <= Shape2D.PREC &
                            unmasked[j])

        return xsects.mask_where(new_mask)
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

        #-----------------------------------------------
        # Define relative points at all the corners
        #-----------------------------------------------
        corners = []
        for corner_pt in self.ptsn:
            corners.append(corner_pt - pt).unit()

        corners = Qube.stack(*corners)

        #------------------------------
        # Construct the average
        #------------------------------
        center = corners.mean(axis=0)

        #---------------------------------------------------------------------
        # The two most widely separated vectors will have the extreme cross
        # products
        #---------------------------------------------------------------------
        crosses = corners.cross(center)
        argmin = crosses.argmin(axis=0)
        argmax = crosses.argmax(axis-0)

        max_indx = Shape2D._meshgrid_for_arg(argmax)
        min_indx = Shape2D._meshgrid_for_arg(argmin)
        return (self.pts[min_indx], self.pts[max_indx])
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
        pts = self.point_at(t)
        side = (t % self.n).as_int()
        pt0 = self.pts[side]
        pt1 = self.pts[side + 1]

        pt0 = pt0.mask_where((side - t).abs() < Shape2D.PREC)
                                            # corner points have no tangent

        return Line(pt0, pt1)
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
        side = (t % self.n).as_int()
        dpt = -self.dpt[side] * self.sign

        pt0 = pt0.mask_where((side - t).abs() < Shape2D.PREC)
                                            # corner points have no normal

        #---------------------------------------------------------------------
        # rotate90 will rotate counterclockwise and so point inward for any
        # polygon defined in a counterclockwise direction. This ensures that
        # all lines will point outward after rotate90.
        #---------------------------------------------------------------------
        return HalfLine(pt0, pt0 + dpt).rotate90()
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

        #----------------------------
        # Convex shapes are easy
        #----------------------------
        if type(arg) == Pair:
            pt = Point(arg)

        if arg.is_convex().all():
            is_subset = []
            for pt in self.pts:
                is_subset.append(pt.is_subset_of(arg))
            is_subset = Qube.stack(*is_subset)
            return is_subset.all()

        #---------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #---------------------------------------------------------------
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

        #---------------------------------------
        # Is superset of, Polygon to Point
        #---------------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            crosses = self.dpts.cross(pt - self.pts3)
            return (crosses * self.signs >= 0).all(axis=0)

        #-------------------------------------
        # Is superset of, Polygon to Line
        #-------------------------------------
        if type(arg) in (Line, HalfLine):
            return Boolean.FALSE

        if type(arg) == Segment:
            line = arg
            return self.is_superset_of(line.pt0) & self.is_superset_of(line.pt1)

        #---------------------------------------
        # Is superset of, Polygon to Circle
        #---------------------------------------
        if type(arg) == Circle:
            circle = arg

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # A polygon is a superset of a circle if...
            # 1. The circle's center is inside the polygon
            # 2. Every point is outside the circle.
            # 3. None of the sides intersects the circle (although they can
            #    touch)
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            circle_center_is_inside = self.is_superset_of(circle.pt0)

            vertex_is_inside = []
            for pt in self.pts:
                vertex_is_inside.append(circle.is_superset_of(pt))
            vertex_is_inside = Qube.stack(*vertex_is_inside)
            vertices_are_outside = ~vertex_is_inside.any(axis=0)

            side_intersects = []
            for seg in self.segs:
                # Second index [1] is masked when circle and segment just touch
                side_intersects.append(seg.intersections(circle)[1].antimask)
            side_intersects = Qube.stack(*side_intersects)
            sides_dont_intersect = ~side_intersects.any(axis=0)

            return (circle_center_is_inside & vertices_are_outside &
                    sides_dont_intersect)

        #---------------------------------------
        # Is superset of, Polygon to Triangle
        #---------------------------------------
        if type(arg) == Triangle:
            triangle = arg

            return (self.is_superset_of(triangle.pt0) &
                    self.is_superset_of(triangle.pt1) &
                    self.is_superset_of(triangle.pt2))

        #----------------------------------------
        # Is superset of, Polygon to Polygon
        #----------------------------------------
        if type(arg) == Polygon:
            polygon = arg

            vertex_is_inside = []
            for pt in polygon.pts:
                vertex_is_inside.append(elf.is_superset_of(pt))
            vertex_is_inside = Qube.stack(*vertex_is_inside)
            return vertex_is_inside.all(axis=0)

        #-----------------------------------------------------------------
        # For other cases, use the method of the other object's class
        #-----------------------------------------------------------------
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
        
        #----------------------------------------------
        # Is disjoint from, Triangle to Polygon
        #----------------------------------------------
        if type(arg) in (Point, Pair):
            return ~self.is_superset_of(arg)

        #----------------------------------------------
        # Is disjoint from, Polygon to Line
        #----------------------------------------------
        if type(arg) == Line:
            line = arg

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # A polygon is disjoint from an infinite line if all points
            # fall on the same side of the line.
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

            diffs = self.ptsn - line.pt0
            sides = diffs.dot(line.perp)
            return (sides >= 0.).all(axis=0) | (sides <= 0.).all(axis=0)

        if type(arg) in (HalfLine, Segment):
            line == arg

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # A polygon is disjoint from a half-line or segment if the full
            # line would be disjoint or if one endpoint is outside the polygon
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

            return (self.is_disjoint_from(Line.as_line(line)) |
                    self.is_disjoin_from(line.pt0))

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

        #--------------------------------
        # Touches, Polygon to Point
        #--------------------------------
        if type(arg) in (Point, Pair):
            pt = Point(arg)

            point_touches = []
            for seg in self.segs:
                point_touches.append(seg.touches(pt))
            point_touches = Qube.stack(*point_touches)
            return point_touches.any(axis=0)

        #--------------------------------
        # Touches, Polygon to Line
        #--------------------------------
        if isinstance(type(arg), Line):
            line = arg

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # True if exactly one line endpoint touches the triangle or if the
            # line touches one corner
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            touchings = []
            for pt in self.pts:
                touchings.append(line.touches(pt))

            #- - - - - - - - - - - - - - - - - - - - - - - - -
            # Count matches on the side but not the corners
            #- - - - - - - - - - - - - - - - - - - - - - - - -
            for seg in self.segs:
                (side_pt, line_pt) = seg.closest(line)
                touches_side = (side_pt - line_pt).norm_sq() <= Shape2D.PREC_SQ
                mask = ((side_pt - seg.pt0).norm_sq() <= Shape2D.PREC_SQ) |
                        (side_pt - seg.pt1).norm_sq() <= Shape2D.PREC_SQ))
                touches_side[mask] = False
                touchings.append(touches_side)

            touchings = Qube.stack(*touchings)
            return (touchings.sum(axis=0) == 1)

        #-------------------------------------
        # Otherwise use the general method
        #-------------------------------------
        return super(Shape2D, self).touches(arg)
    #===========================================================================


#*******************************************************************************



################################################################################
