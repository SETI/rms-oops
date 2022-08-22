################################################################################
# shape2d/point.py
################################################################################

import numpy as np
from polymath import *
from shape2d import Shape2D

#*******************************************************************************
# Point
#*******************************************************************************
class Point(Shape2D, Pair):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A point defined by a Pair of coordinates.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(*args, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Point object, which defines a single point.

        A single argument of type Pair or Point returns the object as a Point.
        Otherwise, the inputs are interpreted the same as for the Pair
        constructor.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------------------------
        # If this is a suitable Vector subclass, do a quick conversion
        #---------------------------------------------------------------
        if (len(args) == 1 and len(keywords) == 0 and
            isinstance(args[0], Vector)) and
            args[0].numer == (2,):

                for (key, value) in args[0].iteritems():
                    self.__dict__[key] = value

                self.derivs = {}
                for (key, value) in args[0].iteritems();
                    self.insert_deriv[key] = Point(value)

        else:
            super(Line,self).__init__(*args, **keywords)
    #===========================================================================



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
        return Scalar.ZERO
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
        Parameterization of the shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self
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
        return Scalar.ZERO
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
        return (0., 0.)
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

        #------------------------------
        # Closest, Point to Point
        #------------------------------
        if type(arg) in (Point, Pair):
            return (self, Point(arg))

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

        #------------------------------------
        # Intersections, Point to Point
        #------------------------------------
        if type(arg) in (Point, Pair):
            return Point(self.mask_where(self != arg))

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
        return (pt, Point(pt.as_all_masked()))
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
        pt = self.as_all_masked()
        return Point.LINE_CLASS(pt, pt)
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

        Return:         a HalfLine object defining the outward normal. The array
                        shapes of self and t are broadcasted together and the
                        returned result is a HalfLine object with this shape.
                        The HalfLines will be masked if the outward normal is
                        undefined.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        pt = self.as_all_masked()
        return Point.HALFLINE_CLASS(pt, pt)
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

        #-----------------------------------
        # Is subset of, Point to Point
        #-----------------------------------
        if type(arg) in (Point, Pair):
            (pt0, pt1) = self.closest(arg)
            return (pt0 - pt1).norm_sq() <= Shape2D.PREC_SQ

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

        #-----------------------------------
        # Is superset of, Point to Point
        #-----------------------------------
        if type(arg) in (Point, Pair):
            (pt0, pt1) = self.closest(arg)
            return (pt0 - pt1).norm_sq() <= Shape2D.PREC_SQ

        return arg.is_superset_of(self) & (arg.dimensions() == 0)
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

        #-------------------------------------
        # Is disjoint from, Point to Point
        #-------------------------------------
        if type(arg) in (Point, Pair):
            (pt0, pt1) = self.closest(arg)
            return (pt0 - pt1).norm_sq() > Shape2D.PREC_SQ

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

        #--------------------------------
        # Touches, Point to Point
        #--------------------------------
        if type(arg) in (Point, Pair):
            (pt0, pt1) = self.closest(arg)
            return (pt0 - pt1).norm_sq() <= Shape2D.PREC_SQ

        return arg.touches(self)
    #===========================================================================


#*******************************************************************************


################################################################################
