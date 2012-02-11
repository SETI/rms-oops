################################################################################
# oops/fov/fov.py: Abstract class FOV (Field-of-View)
#
# 2/2/12 Modified (MRS) - converted to new class names and hierarchy.
################################################################################

import numpy as np

from oops.xarray.all import *

class FOV(object):
    """The FOV (Field of View) abstract class provides a description of the
    geometry of a field of view. 

    The properties of an FOV are defined within a fixed coordinate frame, with
    the positive Z axis oriented near the center of the line of sight. The X and
    Y axes are effectively in the plane of the FOV, with the X-axis oriented
    horizontally and the Y-axis pointed downward. The values for (x,y) are
    implemented using a "pinhole camera" model, in which the z-component has
    unit length. Therefore, at least near the center of the field of view, the
    units of x and y are radians.

    The FOV converts between the actual line of sight vector (x,y,z) and an
    internal coordinate system (ICS) that typically defines a pixel grid. It
    also accommodates any spatial distortion of the field of view. The ICS
    coordinates (u,v) are linear with the grid of pixels and (typically) a unit
    step in (u,v) shifts the position by one pixel. The u-axis points rightward
    (in the direction of increasing sample number) and they V-axis points either
    upward or downward, in the direction of increasing line numbers.

    Although ICS coordinates (u,v) are defined here in units of pixels, the FOV
    concept can be used to describe an arbitrary field of view, consisting of
    detectors of arbitary shape. In this case, (u,v) are simply a convenient set
    of coordinates to use a in frame the describes the layout of detectors on
    the focal plane of the instrument.

    This class may have to be extended if necessary to handle a field of view
    that time-dependency, such as a pushbroom or raster-scanning imager.

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

    def xy_from_uv(self, uv_pair):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v)."""

        pass

    def uv_from_xy(self, xy_pair):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians."""

        pass

    def xy_and_dxy_duv_from_uv(self, uv_pair):
        """Returns a tuple ((x,y), dxy_duv), where the latter is the set of
        partial derivatives of (x,y) with respect to (u,v). These are returned
        as a Pair object of shape [...,2]:
            dxy_duv[...,0] = Pair((dx/du, dx/dv))
            dxy_duv[...,1] = Pair((dy/du, dy/dv))
        """

        pass

########################################################
# Derived methods, to override only if necessary
########################################################

    def area_factor(self, uv_pair):
        """Returns the relative area of a pixel or other sensor at (u,v)
        coordinates, compared to a nominal pixel area.
        """

        # Get the partial derivatives
        (xy_pair, dxy_duv) = self.xy_and_dxy_duv_from_uv(uv_pair)

        dx_du = dxy_duv.vals[...,0,0]
        dx_dv = dxy_duv.vals[...,0,1]
        dy_du = dxy_duv.vals[...,1,0]
        dy_dv = dxy_duv.vals[...,1,1]

        # Construct the cross products
        return Scalar(np.abs(dx_du * dy_dv - dx_dv * dy_du) / self.uv_area)

    # This models the field of view as a pinhole camera
    def los_from_xy(self, xy_pair):
        """Returns a unit Vector3 object pointing in the direction of the
        specified spatial coordinate Pair (x,y). Note that this is the direction
        _opposite_ to that in which the photon is moving.
        """

        # Convert to Pair if necessary
        xy_pair = Pair.as_pair(xy_pair)

        # Fill in the numpy ndarray of vector components
        buffer = np.ones(xy_pair.shape + [3])
        buffer[...,0:2] = xy_pair.vals

        # Convert to Vector3 and return
        return Vector3(buffer)

    def xy_from_los(self, los):
        """Returns the coordinate Pair (x,y) based on a Vector3 object pointing
        in the direction of a line of sight. Lines of sight point outward from
        the camera, near the Z-axis, and are therefore opposite to the direction
        in which a photon is moving. The length of the vector is ignored."""

        # Scale to z=1 and then convert to Pair
        los = Vector3.as_vector3(los)
        return Pair(los.vals[...,0:2] / los.vals[...,2:3])

    def los_from_uv(self, uv_pair):
        """Returns a Vector3 object pointing in the direction of the specified
        by coordinate Pair (u,v). Note that this is the direction _opposite_ to
        that in which the photon is moving.
        """

        return self.los_from_xy(self.xy_from_uv(uv_pair))

    def uv_from_los(self, los):
        """Returns the coordinate Pair (u,v) based on a line of sight Vector3
        object pointing in the direction of a line of sight, i.e., _opposite_ to
        the direction in which the photon is moving. The length of the vector is
        ignored."""

        return self.uv_from_xy(self.xy_from_los(los))

    def uv_is_inside(self, uv_pair):
        """Returns a boolean Scalar indicating True for (u,v) coordinates that
        fall inside the FOV, False otherwise."""

        uv_pair = Pair.as_pair(uv_pair)
        return Scalar((uv_pair.vals[...,0] >= 0.) &
                      (uv_pair.vals[...,1] >= 0.) &
                      (uv_pair.vals[...,0] <= self.uv_shape.vals[0]) &
                      (uv_pair.vals[...,1] <= self.uv_shape.vals[1]))

    def xy_is_inside(self, xy_pair):
        """Returns a boolean Scalar indicating True for (x,y) coordinates that
        fall inside the FOV, False otherwise."""

        return self.uv_is_inside(self.uv_from_xy(xy_pair))

    def los_is_inside(self, los):
        """Returns a boolean Scalar indicating True line of sight vectors that
        fall inside the FOV, False otherwise."""

        return self.uv_is_inside(self.uv_from_los(los))

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
