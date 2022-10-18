################################################################################
# oops/fov/__init__.py: Abstract class FOV (Field-of-View)
################################################################################

import numpy as np
from polymath import Scalar, Pair, Vector3

class FOV(object):
    """The FOV (Field of View) abstract class provides a description of the
    geometry of a field of view.

    The properties of an FOV are defined within a fixed coordinate frame, with
    the positive Z axis oriented near the center of the line of sight. The x and
    y axes are effectively in the plane of the FOV, with the x-axis oriented
    horizontally and the y-axis pointed downward. The values for (x,y) are
    implemented using a "pinhole camera" model, in which the z-component has
    unit length. Therefore, at least near the center of the field of view, the
    units of x and y are radians.

    The FOV converts between the actual line of sight vector (x,y,z) and an
    internal coordinate system (ICS) that typically defines a pixel grid. It
    also accommodates any spatial distortion of the field of view. The ICS
    coordinates (u,v) are linear with the grid of pixels and, typically, a unit
    step in (u,v) shifts the position by one pixel. The u-axis points rightward
    in the default display orientation of a data array, and the v-axis points
    either upward or downward.

    Although ICS coordinates (u,v) are defined here in units of pixels, the FOV
    concept can be used to describe an arbitrary field of view, consisting of
    detectors of arbitrary shape. In this case, (u,v) are simply a convenient
    set of coordinates to use in a frame that describes the layout of detectors
    on the focal plane of the instrument.

    The class also allows for the possibility that the field of view has
    additional dependencies on wavelength, etc. Additional arguments and keyword
    values can be passed through these methods and into the subclass methods.

    Every FOV should have the following attributes:
        uv_los      a Pair defining the (u,v) coordinates of the nominal line of
                    sight.

        uv_scale    a Pair defining the approximate ratios dx/du and dy/dv. For
                    example, if (u,v) are in units of arcseconds, then
                        uv_scale = Pair((pi/180/3600.,pi/180/3600.))
                    Use the sign of the second element to define the direction
                    of increasing V: negative for up, positive for down.

        uv_shape    a Pair defining size of the field of view in pixels. This
                    number can be non-integral if the detector is not composed
                    of a rectangular array of pixels.

        uv_area     the nominal area of a region defined by unit steps in (u,v),
                    e.g., the size of a pixel in steradians.
    """

    # Override this class attribute to False for FOV subclasses that have
    # time-dependence
    IS_TIME_INDEPENDENT = True

    ############################################################################
    # Methods to be defined for each FOV subclass
    ############################################################################

    def __init__(self):
        """The constructor."""

        pass

    #===========================================================================
    def xy_from_uvt(self, uv_pair, tfrac=0.5, time=None, derivs=False,
                          **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        raise NotImplementedError('xy_from_uvt() is not implemented')

    #===========================================================================
    def uv_from_xyt(self, xy_pair, tfrac=0.5, time=None, derivs=False,
                          **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        raise NotImplementedError('uv_from_xyt() is not implemented')

    ############################################################################
    # Derived methods, to override only if necessary
    ############################################################################

    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v),
        assuming the FOV is time-independent.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        assert self.IS_TIME_INDEPENDENT
        return self.xy_from_uvt(uv_pair, derivs=derivs, **keywords)

    #===========================================================================
    def uv_from_xy(self, xy_pair, derivs=False, **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates,
        assuming the FOV is time-independent.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        assert self.IS_TIME_INDEPENDENT
        return self.uv_from_xyt(xy_pair, derivs=derivs, **keywords)

    #===========================================================================
    def area_factor(self, uv_pair, tfrac=0.5, time=None, **keywords):
        """The relative area of a pixel or other sensor at (u,v) at the
        specified time (although any dependence on tfrac should be very small).

        Results are scaled to the nominal pixel area.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            uv_pair     Pair of (u,v) coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.

        Return:         relative area of the pixel at (u,v), as a Scalar.
        """

        # Prepare for the partial derivatives
        uv_pair = Pair.as_pair(uv_pair).wod
        uv_pair = uv_pair.with_deriv('uv', Pair.IDENTITY, 'insert')
        xy_pair = self.xy_from_uvt(uv_pair, tfrac, time, derivs=True,
                                   **keywords)

        dx_du = xy_pair.d_duv.vals[...,0,0]
        dx_dv = xy_pair.d_duv.vals[...,0,1]
        dy_du = xy_pair.d_duv.vals[...,1,0]
        dy_dv = xy_pair.d_duv.vals[...,1,1]

        # Construct the cross products
        return Scalar(np.abs(dx_du * dy_dv - dx_dv * dy_du) / self.uv_area,
                      xy_pair.mask)

    #===========================================================================
    def los_from_xy(self, xy_pair, derivs=False):
        """The unit line-of-sight vector for camera coordinates (x,y).

        Note that this vector points in the direction _opposite_ to the path
        of arriving photons.

        Default behavior it to model the field of view as a pinhole camera.

        Input:
            xy_pair     Pairs of (x,y) coordinates.
            derivs      True to propagate any derivatives in (x,y) forward
                        into the line-of-sight vector.

        Return:         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
        """

        # Convert to Pair if necessary
        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)

        # In the pinhole camera model, the z-component is always 1
        (x,y) = Pair.to_scalars(xy_pair)
        return Vector3.from_scalars(x,y,1.).unit(derivs)

    #===========================================================================
    def xy_from_los(self, los, derivs=False):
        """Camera frame coordinates (x,y) given a line of sight.

        Lines of sight point outward from the camera, near the Z-axis, and are
        therefore opposite to the direction in which a photon is moving. The
        length of the vector is ignored.

        Input:
            los         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
            derivs      True to propagate any derivatives in (x,y) forward
                        into the line-of-sight vector.

        Return:         Pair of (x,y) coordinates in the camera's frame.
        """

        # Scale to z=1 and then convert to Pair
        los = Vector3.as_vector3(los, recursive=derivs)
        z = los.to_scalar(2)
        los = los / z

        return los.to_pair((0,1))

    #===========================================================================
    def los_from_uvt(self, uv_pair, tfrac=0.5, time=None, derivs=False,
                           **keywords):
        """The line of sight vector in the camera's frame, given FOV
        coordinates (u,v) at the specified time.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            uv_pair     Pair of (u,v) coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.

        Return:         Vector3 direction of the line of sight in the camera's
                        frame.
        """

        xy_pair = self.xy_from_uvt(uv_pair, tfrac, time, derivs=derivs,
                                   **keywords)
        return self.los_from_xy(xy_pair, derivs=derivs)

    #===========================================================================
    def los_from_uv(self, uv_pair, derivs=False, **keywords):
        """The line of sight vector given FOV coordinates (u,v), assuming this
        FOV is time-independent.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            uv_pair     Pair of (u,v) coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.

        Return:         Vector3 direction of the line of sight in the camera's
                        frame.
        """

        xy_pair = self.xy_from_uv(uv_pair, derivs=derivs, **keywords)
        return self.los_from_xy(xy_pair, derivs=derivs)

    #===========================================================================
    def uv_from_los_t(self, los, tfrac=0.5, time=None, derivs=False,
                            **keywords):
        """The FOV coordinates (u,v) given a line of sight vector in the
        camera's frame at the specified time.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            los         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.

        Return:         Pair of (u,v) coordinates in the FOV.
        """

        xy_pair = self.xy_from_los(los, derivs=derivs)
        return self.uv_from_xyt(xy_pair, 
                            tfrac=tfrac, time=time, derivs=derivs, **keywords)

    #===========================================================================
    def uv_from_los(self, los, derivs=False, **keywords):
        """The FOV coordinates (u,v) given a line of sight vector, assuming the
        FOV is time-independent.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Additional parameters that might affect the transform can be included
        as keyword arguments.

        Input:
            los         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.

        Return:         Pair of (u,v) coordinates in the FOV.
        """

        assert self.IS_TIME_INDEPENDENT
        return self.uv_from_los_t(los, derivs=derivs, **keywords)

    #===========================================================================
    def uv_is_outside(self, uv_pair, tfrac=0.5, time=None, inclusive=True,
                            uv_min=None, uv_max=None):
        """A Boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpret them as
                        outside.
            uv_min      an integer Pair representing the lower (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
            uv_max      an integer Pair representing the upper (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.

        Return:         a Boolean indicating True where the point is outside the
                        FOV.
        """

        # Interpret the (u,v) coordinates
        uv_pair = Pair.as_pair(uv_pair, recursive=False)
        (u,v) = uv_pair.to_scalars()

        # Fill in the corners
        if uv_min is None:
            uv_min = Pair.INT00

        if uv_max is None:
            uv_max = self.uv_shape

        (umin, vmin) = uv_min.vals
        (umax, vmax) = uv_max.vals

        # Create the mask
        if inclusive:
            result = (u < umin) | ( v < vmin) | (u > umax) | (v > vmax)
        else:
            result = (u < umin) | (v < vmin) | (u >= umax) | (v >= vmax)

        return result

    #===========================================================================
    def u_or_v_is_outside(self, uv_coord, uv_index, inclusive=True,
                                          uv_min=None, uv_max=None):
        """A Boolean mask identifying coordinates outside the FOV.

        Input:
            uv_coord    a Scalar of u-coordinates or v-coordinates.
            uv_index    0 to test u-coordinates; 1 to test v-coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.
            uv_min      an integer Pair representing the lower (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
            uv_max      an integer Pair representing the upper (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.

        Return:         a Boolean indicating True where the point is outside
                        the FOV.
        """

        # Interpret the (u,v) coordinates
        uv_coord = Scalar.to_scalar(uv_coord)
        shape = self.uv_shape.vals

        # Fill in the corners
        if uv_min is None:
            uv_min = Pair.INT00

        if uv_max is None:
            uv_max = self.uv_shape

        (umin, vmin) = uv_min.vals
        (umax, vmax) = uv_max.vals

        # Create the mask
        if inclusive:
            result = (uv_coord < 0) | (uv_coord > shape[uv_index])
        else:
            result = (uv_coord < 0) | (uv_coord >= shape[uv_index])

        return result

    #===========================================================================
    def xy_is_outside(self, xy_pair, tfrac=0.5, time=None, inclusive=True,
                            uv_min=None, uv_max=None, **keywords):
        """A Boolean mask identifying coordinates outside the FOV.

        Input:
            xy_pair     a Pair of (x,y) coordinates, assuming z == 1.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.
            uv_min      an integer Pair representing the lower (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
            uv_max      an integer Pair representing the upper (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
        """

        uv = self.uv_from_xyt(xy_pair, tfrac=tfrac, derivs=False, **keywords)
        return self.uv_is_outside(uv, inclusive, uv_min, uv_max)

    #===========================================================================
    def los_is_outside(self, los, tfrac=0.5, time=None, inclusive=True,
                             uv_min=None, uv_max=None, **keywords):
        """A Boolean mask identifying lines of sight outside the FOV.

        Input:
            los         an outward line-of-sight vector.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        Scalar of optional absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.
            uv_min      an integer Pair representing the lower (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
            uv_max      an integer Pair representing the upper (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
        """

        xy = self.xy_from_los(derivs=False)
        return self.xy_is_outside(xy, tfrac, inclusive=inclusive,
                                  uv_min=uv_min, uv_max=uv_max, **keywords)

    #===========================================================================
    def nearest_uv(self, uv_pair, remask=False):
        """The closest (u,v) coordinates inside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            remask      True to mask the points outside the boundary.

        Return:         a new Pair of (u,v) coordinates.
        """

        clipped = Pair.as_pair(uv_pair).copy(readonly=False, recursive=False)
        clipped.vals[...,0] = clipped.vals[...,0].clip(0, self.uv_shape.vals[0])
        clipped.vals[...,1] = clipped.vals[...,1].clip(0, self.uv_shape.vals[1])

        if remask:
            return Pair(clipped, uv_pair.mask | (clipped != uv_pair))
        else:
            return clipped

    ############################################################################
    # Properties and methods to support body inventories
    #
    # These might need to be overridden for FOV subclasses that are not
    # rectangular.
    ############################################################################

    def center_xy(self, tfrac=0.5, time=None):
        """The (x,y) coordinate pair at the center of the (u,v) center of the
        FOV in the camera's frame at the specified time.
        """

        if hasattr(self, 'center_xy_filled'):
            return self.center_xy_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.center_xy_filled = self.xy_from_uvt(self.uv_shape/2.)
            return self.center_xy_filled

        return self.xy_from_uvt(self.uv_shape/2., tfrac, time)

    #===========================================================================
    def center_los(self, tfrac=0.5, time=None):
        """The unit line of sight defining the (u,v) center of the FOV in the
        camera's frame at the specified time.
        """

        if hasattr(self, 'center_los_filled'):
            return self.center_los_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.center_los_filled = self.los_from_xy(self.center_xy()).unit()
            return self.center_los_filled

        return self.los_from_xy(self.center_xy(tfrac, time)).unit()

    #===========================================================================
    @property
    def center_dlos_duv(self):
        """The line of sight derivative matrix dlos/d(u,v) at the FOV center.

        Note that any time-dependence is ignored.
        """

        if not hasattr(self, 'center_dlos_duv_filled'):
            center_uv = self.uv_shape/2.
            center_uv.insert_deriv('uv', Pair.IDENTITY)
            los = self.los_from_uvt(center_uv, derivs=True)
            self.center_dlos_duv_filled = los.d_duv

        return self.center_dlos_duv_filled

    #===========================================================================
    @property
    def outer_radius(self):
        """The radius in radians of a circle circumscribing the entire FOV.

        Note that any time-dependence of the FOV is ignored.
        """

        if not hasattr(self, 'outer_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]
            uv_corners = Pair([(0.,0.), (0.,vmax), (umax,0.), (umax,vmax)])

            seps = self.center_los().sep(self.los_from_uvt(uv_corners))
            self.outer_radius_filled = seps.max()

        return self.outer_radius_filled

    #===========================================================================
    @property
    def inner_radius(self):
        """The radius in radians of a circle entirely enclosed within the FOV.

        Note that any time-dependence of the FOV is ignored.
        """

        if not hasattr(self, 'inner_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]
            umid = umax/2.
            vmid = vmax/2.

            uv_edges = Pair([(0.,vmid), (umax,vmid), (umid,0.), (umid,vmax)])

            seps = self.center_los().sep(self.los_from_uvt(uv_edges))
            self.inner_radius_filled = seps.min()

        return self.inner_radius_filled

    #===========================================================================
    def corner00_xy(self, tfrac=0.5, time=None):
        """The (x,y) Pair at (u,v) coordinates (0,0) at a specified time."""

        if hasattr(self, 'corner00_filled'):
            return self.corner00_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.corner00_filled = self.xy_from_uvt(Scalar.ZEROS)
            return self.corner00_filled

        return self.xy_from_uvt(Scalar.ZEROS, tfrac, time)

    #===========================================================================
    def corner01_xy(self, tfrac=0.5, time=None):
        """The (x,y) Pair at (u,v) coordinates (0,v_max) at a specified time."""

        if hasattr(self, 'corner01_filled'):
            return self.corner01_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.corner01_filled = self.xy_from_uvt([0,self.uv_shape[1]])
            return self.corner01_filled

        return self.xy_from_uvt([0,self.uv_shape[1]], tfrac, time)

    #===========================================================================
    def corner10_xy(self, tfrac=0.5, time=None):
        """The (x,y) Pair at (u,v) coordinates (u_max,0) at a specified time."""

        if hasattr(self, 'corner10_filled'):
            return self.corner10_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.corner10_filled = self.xy_from_uvt([self.uv_shape[0],0])
            return self.corner10_filled

        return self.xy_from_uvt(Scalar.ZEROS, tfrac, time)

    #===========================================================================
    def corner11_xy(self, tfrac=0.5, time=None):
        """The (x,y) Pair at (u,v) coordinates (u_max,v_max) at a specified
        time.
        """

        if hasattr(self, 'corner11_filled'):
            return self.corner11_filled

        if self.IS_TIME_INDEPENDENT or (tfrac is not None and
                                        (Scalar(tfrac) == 0.5).all()):

            self.corner11_filled = self.xy_from_uvt(self.uv_shape)
            return self.corner11_filled

        return self.xy_from_uvt(self.uv_shape, tfrac, time)

    #===========================================================================
    def sphere_falls_inside(self, center, radius, tfrac=0.5, time=None,
                                  border=0.):
        """True if any piece of sphere falls inside a field of view.

        Input:
            center      the apparent location of the center of the sphere in the
                        internal coordinate frame of the FOV.
            radius      the radius of the sphere.
            tfrac       Scalar of fractional times during the exposure, where
                        tfrac=0 at the beginning and 1 at the end. Default is
                        0.5.
            time        optional Scalar of absolute time in seconds. Only one of
                        tfrac and time can be specified; the other must be None.
            border      an optional angular extension to the field of view, in
                        radians, to allow for pointing uncertainties.
        """

        # Perform quick tests based on the separation angles
        sphere_center_los = Vector3.as_vector3(center, recursive=False)

        radius_angle = (radius / sphere_center_los.norm()).arcsin()
        center_los = self.center_los(tfrac=tfrac, time=time)
        center_sep = center_los.sep(sphere_center_los)

        if center_sep > self.outer_radius + border + radius_angle:
            return False
        if center_sep <= self.inner_radius + border + radius_angle:
            return True

        # Find the point on the image that falls closest to the center of the
        # sphere
        sphere_center_uv = self.uv_from_los_t(sphere_center_los, tfrac, time)
        nearest_fov_uv  = self.nearest_uv(sphere_center_uv)
        nearest_fov_los = self.los_from_uvt(nearest_fov_uv, tfrac, time)

        # Allow for the border region when returning True or False
        return nearest_fov_los.sep(sphere_center_los) <= radius_angle + border

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_FOV(unittest.TestCase):

    def runTest(self):

        # Fully tested by Flat.py

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
