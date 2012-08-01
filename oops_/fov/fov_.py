################################################################################
# oops_/fov/fov.py: Abstract class FOV (Field-of-View)
#
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
# 2/22/12 MRS - Revised the handling of derivatives.
# 3/9/12 MRS - Added the "extras" argument to support additional Observation
#   subclasses
################################################################################

import numpy as np

from oops_.array.all import *

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
    detectors of arbitary shape. In this case, (u,v) are simply a convenient set
    of coordinates to use in a frame that describes the layout of detectors on
    the focal plane of the instrument.

    The class also allows for the possibility that the field of view has
    additional dependencies on wavelength, etc. Most functions support a tuple
    of arguments called extras to contain these additional parameters.

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

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        pass

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        pass

########################################################
# Derived methods, to override only if necessary
########################################################

    def area_factor(self, uv_pair, extras=()):
        """Returns the relative area of a pixel or other sensor at (u,v)
        coordinates, compared to a nominal pixel area.

        Additional parameters that might affect the transform can be included
        in the extras argument.
        """

        # Get the partial derivatives
        xy = self.xy_from_uv(uv_pair, extras, derivs=True)

        dx_du = xy.d_duv.vals[...,0,0]
        dx_dv = xy.d_duv.vals[...,0,1]
        dy_du = xy.d_duv.vals[...,1,0]
        dy_dv = xy.d_duv.vals[...,1,1]

        # Construct the cross products
        return Scalar(np.abs(dx_du * dy_dv - dx_dv * dy_du) / self.uv_area,
                      xy.mask)

    # This models the field of view as a pinhole camera
    def los_from_xy(self, xy_pair, derivs=False):
        """Returns a unit Vector3 object pointing in the direction of the line
        of sight of the specified coordinate Pair (x,y). Note that this is the
        direction _opposite_ to that in which the photon is moving.

        If derivs is True, then the returned Vector3 has a subfield "d_dxy",
        which contains the derivatives as a MatrixN with item shape [3,2].
        """

        # Convert to Pair if necessary
        xy_pair = Pair.as_pair(xy_pair)

        # Fill in the numpy ndarray of vector components
        vals = np.ones(xy_pair.shape + [3])
        vals[...,0:2] = xy_pair.vals

        # Convert to a unit Vector3
        los = Vector3(vals, xy_pair.mask).unit()

        # Attach the derivatives if necessary
        if derivs:
            # los_x = x / sqrt(1 + x**2 + y**2)
            # los_y = y / sqrt(1 + x**2 + y**2)
            # los_z = 1 / sqrt(1 + x**2 + y**2)
            #
            # dlos/d(x,y) = ([ 1+y**2,    -xy],
            #                [    -xy, 1+x**2],
            #                [     -x,     -y]) * (1 + x**2 + y**2)**(-3/2)

            x = xy_pair.vals[...,0]
            y = xy_pair.vals[...,1]

            dlos_dxy_vals = np.empty(los.shape + [3,2])
            dlos_dxy_vals[...,0,0] = 1 + y**2
            dlos_dxy_vals[...,0,1] = -x * y
            dlos_dxy_vals[...,1,0] = dlos_dxy_vals[...,0,1]
            dlos_dxy_vals[...,1,1] = 1 + x**2
            dlos_dxy_vals[...,2,:] = -xy_pair.vals[...,:]

            normalize = ((dlos_dxy_vals[...,0,0] +
                          dlos_dxy_vals[...,1,1] - 1)**(-1.5))
            dlos_dxy_vals *= normalize[..., np.newaxis, np.newaxis]

            los.insert_subfield("d_dxy", MatrixN(dlos_dxy_vals, xy_pair.mask))

        return los

    def xy_from_los(self, los, derivs=False):
        """Returns the coordinate Pair (x,y) based on a Vector3 object pointing
        in the direction of a line of sight. Lines of sight point outward from
        the camera, near the Z-axis, and are therefore opposite to the direction
        in which a photon is moving. The length of the vector is ignored.

        If derivs is True, then the Pair returned has a subfield "d_dlos", which
        contains the derivatives d(x,y)/d(los) as a MatrixN with item shape
        [2,3].
        """

        # Scale to z=1 and then convert to Pair
        los = Vector3.as_vector3(los)
        xy = Pair(los.vals[...,0:2] / los.vals[...,2:3], los.mask)

        # Construct the derivatives if necessary
        if derivs:
            # x = los_x / los_z
            # y = los_y / los_z
            #
            # dx/dlos_x = 1 / los_z
            # dx/dlos_y = 0
            # dx/dlos_z = -los_x / (los_z**2)
            # dy/dlos_x = 0
            # dy/dlos_y = 1 / los_z
            # dy/dlos_z = -los_y / (los_z**2)
    
            dxy_dlos_vals = np.zeros(los.shape + [2,3])
            dxy_dlos_vals[...,0,0] =  los_vals[...,2]
            dxy_dlos_vals[...,0,2] = -los_vals[...,0]
            dxy_dlos_vals[...,1,1] =  los_vals[...,2]
            dxy_dlos_vals[...,1,2] = -los_vals[...,0]
            dxy_dlos_vals /= los_vals[...,2]

            xy.insert_subfield("d_dlos", MatrixN(dxy_dlos_vals, los.mask))

        return xy

    def los_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns the line of sight (los) as a Vector3 object. The los points
        in the direction specified by coordinate Pair (u,v). Note that this is
        the direction _opposite_ to that in which the photon is moving.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the Vector3 returned has a subfield "d_duv",
        which contains the partial derivatives d(los)/d(u,v) as a MatrixN with
        item shape [3,2]. 
        """

        xy_pair = self.xy_from_uv(uv_pair, extras, derivs)
        los = self.los_from_xy(xy_pair, derivs)

        if derivs:
            los.insert_subfield("d_duv", los.d_dxy * xy_pair.d_duv)
            los.delete_subfield("d_dxy")

        return los

    def uv_from_los(self, los, extras=(), derivs=False):
        """Returns the coordinate Pair (u,v) based on a line of sight Vector3
        pointing in the direction of a line of sight, i.e., _opposite_ to
        the direction in which the photon is moving. The length of the vector is
        ignored.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then Pair return has a subarrray "d_dlos", which
        contains the partial derivatives d(u,v)/d(los) as a MatrixN with item
        shape [2,3]. 
        """

        xy = self.xy_from_los(los, derivs)
        uv = self.uv_from_xy(xy, extras, derivs)

        if derivs:
            uv.insert_subfield("d_dlos", uv.d_dxy * xy.d_dlos)

        return uv

    def uv_is_inside(self, uv_pair, inclusive=True):
        """Returns a boolean NumPy array identifying which coordinates fall
        inside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a boolean NumPy array indicating True where the point is
                        inside the FOV.
        """

        uv_pair = Pair.as_pair(uv_pair)
        if inclusive:
            return ((uv_pair.vals[...,0] >= 0) &
                    (uv_pair.vals[...,1] >= 0) &
                    (uv_pair.vals[...,0] <= self.uv_shape.vals[0]) &
                    (uv_pair.vals[...,1] <= self.uv_shape.vals[1]))
        else:
            return ((uv_pair.vals[...,0] >= 0) &
                    (uv_pair.vals[...,1] >= 0) &
                    (uv_pair.vals[...,0] < self.uv_shape.vals[0]) &
                    (uv_pair.vals[...,1] < self.uv_shape.vals[1]))

    def u_or_v_is_inside(self, uv_coord, uv_index, inclusive=True):
        """Returns a boolean NumPy array identifying which u-coordinates fall
        inside the FOV.

        Input:
            uv_coord    a Scalar of u-coordinates or v-coordinates.
            uv_index    0 to test u-coordinates; 1 to test v-coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.

        Return:         a boolean NumPy array indicating True where the point is
                        inside the FOV.
        """

        uv_coord = Scalar.as_scalar(uv_coord)
        if inclusive:
            return ((u_coord.vals >= 0) &
                    (u_coord.vals <= self.uv_shape.vals[uv_index]))
        else:
            return ((uv_pair.vals >= 0) &
                    (uv_pair.vals < self.uv_shape.vals[uv_index]))

    def xy_is_inside(self, xy_pair, extras=()):
        """Returns a boolean NumPy array indicating True for (x,y) coordinates
        that fall inside the FOV, False otherwise."""

        return self.uv_is_inside(self.uv_from_xy(xy_pair, extras),
                                 inclusive=True)

    def los_is_inside(self, los, extras=()):
        """Returns a boolean NumPy array indicating True for line of sight
        vectors that fall inside the FOV, False otherwise."""

        return self.uv_is_inside(self.uv_from_los(los, extras), inclusive=True)

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
