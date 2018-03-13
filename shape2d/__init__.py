################################################################################
# shape2d/__init__.py
################################################################################

import numpy as np
from polymath import *

class Shape2D(object):
    """Abstract class for 2D shapes."""

    PREC = 1.e-15       # precision required in tests for equality
    PREC_SQ = PREC**2

################################################################################
# Method prototypes
################################################################################

    def dimensions(self):
        """The Scalar dimension of this object: 0 for a point; 1 for a line; 2
        for a shape object that has nonzero area."""

        raise NotImplementedError('method dimensions() is not implemented')

    def is_convex(self):
        """Boolean True if the shape is convex."""

        raise NotImplementedError('method is_convex() is not implemented')

    def point_at(t):
        """Parameterization of the shape."""

        raise NotImplementedError('method point_at() is not implemented')

    def param_at(pt):
        """Parameter at a point, which is assumed to fall on the edge of this
        object.

        What happens when the point does not fall on the shape is undetermined.
        """

        raise NotImplementedError('method param_at() is not implemented')

    def param_limits(self):
        """Parameter limits to define the shape."""

        raise NotImplementedError('method param_limits() is not implemented')

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

        raise NotImplementedError('method closest() is not implemented')

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

        raise NotImplementedError('method intersections() is not implemented')

    def tangents_from(self, pt):
        """The two Points where this Shape2D object is tangent to a line from
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

        raise NotImplementedError('method tangents_from() is not implemented')

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

        raise NotImplementedError('method tangent_at() is not implemented')

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

        raise NotImplementedError('method normal_at() is not implemented')

    def is_subset_of(self, arg):
        """True if this object is as subset of (i.e., is entirely contained by)
        the given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a subset of the given
                        shape.
        """

        raise NotImplementedError('method is_subset_of() is not implemented')

    def is_superset_of(self, arg):
        """True if this object is as superset of (i.e., entirely contains) the
        given Shape2D object.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is a superset of the given
                        shape.
        """

        raise NotImplementedError('method is_superset_of() is not implemented')

    def is_disjoint_from(self, arg):
        """True if the this object and the given Shape2D object are disjoint
        (i.e., do not touch or overlap).

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if this shape is disjoing from the given
                        shape.
        """

        # This is a general method although it may be slow

        # Two object are disjoint if...
        # 1. The second object is not a subset of the first
        # 1. The second object is not a superset of the first
        # 2. The objects do not intersect

        arg_is_superset = self.is_superset_of(arg)
        arg_is_subset = self.is_subset_of(arg)
        arg_intersects = self.intersections(arg).any(axis=0)

        return ~(arg_is_superset | arg_is_subset | arg_intersects)

    def touches(self, arg):
        """True if the this object and the given Shape2D touch but do not share
        any common interior points.

        Input:
            self        this shape.
            arg         another Shape2D object.

        Return:         Boolean True if the shapes touch but share no common
                        interior points.
        """

        # This is a general method although it may be slow
        # NOTE: It only applies to 2D objects

        # Two objects touch if...
        # 1. The second object is not a subset of the first
        # 1. The second object is not a superset of the first
        # 2. The objects intersect exactly once

        if (self.dimensions != 2).any() or (arg.dimensions != 2).any():
            raise ValueError('general touches() method is only implemented ' +
                             'for 2-D shapes')

        arg_is_superset = self.is_superset_of(arg)
        arg_is_subset = self.is_subset_of(arg)
        arg_doesnt_intersect_once = self.intersections(arg).sum(axis=0) != 1

        return ~(arg_is_superset | arg_is_subset | arg_doesnt_intersect_once)

################################################################################
# Support methods
################################################################################

    @staticmethod
    def _meshgrid_for_arg(arg):
        """Meshgrid to convert the results of argmin(axis=0) in an index."""

        axes = [arg.values]
        for k in arg.shape[::-1]:
            axes.append(range(k))

        return np.meshgrid(axes)

    @staticmethod
    def _closest_of_pairings(shape1_pts, shape2_pts)
        """Select the pairings of points on shape1 and shape2 that are closest.
        """

        if type(shape1_pts) == list:
            shape1_pts = Qube.stack(*shape1_pts)

        if type(shape2_pts) == list:
            shape2_pts = Qube.stack(*shape2_pts)

        norm_sq = (shape2_pts - shape1_pts).norm_sq()
        argmin = norm_sq.argmin(axis=0)
        indx = Shape2D._meshgrid_for_arg(argmin)

        return (shape1_pts[indx], shape2_pts[indx])

################################################################################
