################################################################################
# oops/fov_/fov.py: Abstract class FOV (Field-of-View)
################################################################################

import numpy as np
from polymath import *

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

########################################################
# Methods to be defined for each FOV subclass
########################################################

    def __init__(self):
        """A constructor."""

        pass

    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        pass

    def uv_from_xy(self, xy_pair, derivs=False, **keywords):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        pass

########################################################
# Derived methods, to override only if necessary
########################################################

    def area_factor(self, uv_pair, **keywords):
        """The relative area of a pixel or other sensor at (u,v).

        Results are scaled to the nominal pixel area.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        # Prepare for the partial derivatives
        uv_pair = Pair.as_pair(uv_pair).without_derivs()
        uv_pair = uv_pair.with_deriv('uv', Pair.IDENTITY, 'insert')
        xy_pair = self.xy_from_uv(uv_pair, derivs=True, **keywords)

        dx_du = xy_pair.d_duv.vals[...,0,0]
        dx_dv = xy_pair.d_duv.vals[...,0,1]
        dy_du = xy_pair.d_duv.vals[...,1,0]
        dy_dv = xy_pair.d_duv.vals[...,1,1]

        # Construct the cross products
        return Scalar(np.abs(dx_du * dy_dv - dx_dv * dy_du) / self.uv_area,
                      xy_pair.mask)

    # This models the field of view as a pinhole camera
    def los_from_xy(self, xy_pair, derivs=False):
        """Return the unit line-of-sight vector for camera coordinates (x,y).

        Note that this is vector points in the direction _opposite_ to the path
        of arriving photons.

        If derivs is True, then derivatives in (x,y) get propagated forward
        into the components of the line-of-sight vector.
        """

        # Convert to Pair if necessary
        xy_pair = Pair.as_pair(xy_pair, derivs)

        # In the pinhole camera model, the z-component is always 1
        (x,y) = Pair.to_scalars(xy_pair)
        return Vector3.from_scalars(x,y,1.).unit(derivs)

    def xy_from_los(self, los, derivs=False):
        """Return camera frame coordinates (x,y) given a line of sight.

        Lines of sight point outward from the camera, near the Z-axis, and are
        therefore opposite to the direction in which a photon is moving. The
        length of the vector is ignored.

        If derivs is True, then derivatives in the components of the line of
        sight get propagated forward into the components of the (x,y)
        coordinates.
        """

        # Scale to z=1 and then convert to Pair
        los = Vector3.as_vector3(los, derivs)
        z = los.to_scalar(2)
        los = los / z

        return los.to_pair((0,1))

    def los_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return the line of sight vector given FOV coordinates (u,v).

        The los points  the direction specified by coordinate Pair (u,v). Note
        that this is the direction _opposite_ to that of the arriving photon.

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        xy_pair = self.xy_from_uv(uv_pair, derivs, **keywords)
        return self.los_from_xy(xy_pair, derivs)

    def uv_from_los(self, los, derivs=False, **keywords):
        """Return FOV coordinates (u,v) given a line of sight vector.

        The los points  the direction specified by coordinate Pair (u,v). Note
        that this is the direction _opposite_ to that of the arriving photon.

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        xy_pair = self.xy_from_los(los, derivs)
        return self.uv_from_xy(xy_pair, derivs, **keywords)

    def uv_is_outside(self, uv_pair, inclusive=True):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a Boolean indicating True where the point is outside the
                        FOV.
        """

        uv_pair = Pair.as_pair(uv_pair)
        (u,v) = uv_pair.to_scalars()
        (umax, vmax) = self.uv_shape.values

        if inclusive:
            result = (u < 0) | ( v < 0) | (u > umax) | (v > vmax)
        else:
            result = (u < 0) | (v < 0) | (u >= umax) | (v >= vmax)

        if isinstance(result, Qube):
            return result.values        # Convert to NumPy
        else:
            return result               # bool

    def u_or_v_is_outside(self, uv_coord, uv_index, inclusive=True):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_coord    a Scalar of u-coordinates or v-coordinates.
            uv_index    0 to test u-coordinates; 1 to test v-coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """

        uv_coord = Scalar.to_scalar(uv_coord)
        shape = self.uv_shape.values
        if inclusive:
            result = (uv_coord < 0) | (uv_coord > shape[uv_index])
        else:
            result = (uv_coord < 0) | (uv_coord >= shape[uv_index])

        if isinstance(result, Qube):
            return result.values        # Convert to NumPy
        else:
            return result               # bool

    def nearest_uv(self, uv_pair, remask=False):
        """Return the closest (u,v) coordinates inside the FOV.

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

    def xy_is_outside(self, xy_pair, inclusive=True, **keywords):
        """Return a boolean mask identifying coordinates outside the FOV.
        """

        uv = self.uv_from_xy(xy_pair, derivs=False, **keywords)
        return self.uv_is_outside(uv, inclusive)

    def los_is_outside(self, los, inclusive=True, **keywords):
        """Return a boolean mask identifying lines of sight outside the FOV.
        """

        xy = self.xy_from_los(derivs=False)
        return self.xy_is_outside(xy, inclusive, **keywords)

################################################################################
# Properties and methods to support body inventories
#
# These might need to be overridden for FOV subclasses that are not rectangular.
################################################################################

    @property
    def center_xy(self):
        """The (x,y) coordinate pair at the center of the FOV.
        """

        if not hasattr(self, 'center_xy_filled'):
            self.center_xy_filled = self.xy_from_uv(self.uv_shape/2.)

        return self.center_xy_filled

    @property
    def center_los(self):
        """The unit line of sight defining the (u,v) center of the FOV."""

        if not hasattr(self, 'center_los_filled'):
            self.center_los_filled = self.los_from_xy(self.center_xy).unit()

        return self.center_los_filled

    @property
    def center_dlos_duv(self):
        """The line of sight derivative matrix dlos/d(u,v) at the FOV center.
        """

        if not hasattr(self, 'center_dlos_duv_filled'):
            center_uv = Pair(self.uv_shape/2.)
            center_uv.insert_deriv('uv', Pair.IDENTITY)

            los = self.los_from_uv(center_uv, derivs=True)
            self.center_dlos_duv_filled = los.d_duv

        return self.center_dlos_duv_filled

    @property
    def outer_radius(self):
        """The radius in radians of a circle circumscribing the entire FOV.
        """

        if not hasattr(self, 'outer_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]

            uv_corners = Pair([(0.,0.), (0.,vmax), (umax,0.), (umax,vmax)])

            seps = self.center_los.sep(self.los_from_uv(uv_corners))
            self.outer_radius_filled = seps.max()

        return self.outer_radius_filled

    @property
    def inner_radius(self):
        """The radius in radians of a circle entirely enclosed within the FOV.
        """

        if not hasattr(self, 'inner_radius_filled'):
            umax = self.uv_shape.vals[0]
            vmax = self.uv_shape.vals[1]
            umid = umax / 2.
            vmid = vmax / 2.

            uv_edges = Pair([(0.,vmid), (umax,vmid), (umid,0.), (umid,vmax)])

            seps = self.center_los.sep(self.los_from_uv(uv_edges))
            self.inner_radius_filled = seps.min()

        return self.inner_radius_filled

    def sphere_falls_inside(self, center, radius, border=0.):
        """Return True if any piece of sphere falls inside a field of view.

        Input:
            center      the apparent location of the center of the sphere in the
                        internal coordinate frame of the FOV.
            radius      the radius of the spheres.
            border      an optional angular extension to the field of view, in
                        radians, to allow for pointing uncertainties
        """

        # Perform quick tests based on the separation angles
        sphere_center_los = Vector3.as_vector3(center, recursive=False)

        scaled_radius = radius / sphere_center_los.norm()
        radius_angle = scaled_radius.arcsin()
        center_sep = self.center_los.sep(sphere_center_los)

        if center_sep > self.outer_radius + border + radius_angle: return False
        if center_sep < self.inner_radius + border + radius_angle: return True

        # Find the point on the image that falls closest to the center of the
        # sphere
        sphere_center_uv = self.uv_from_los(sphere_center_los)
        nearest_fov_uv  = self.nearest_uv(sphere_center_uv)
        nearest_fov_los = self.los_from_uv(nearest_fov_uv)

        # Allow for the border region when returning True or False
        return nearest_fov_los.sep(sphere_center_los) < radius_angle + border

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
