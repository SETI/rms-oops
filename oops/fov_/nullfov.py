################################################################################
# oops/fov_/flatfov.py: Flat subclass of class FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

class NullFOV(FOV):
    """NullFOV is a subclass of FOV that describes an instrument with no field
    of view, e.g., an in situ instrument.
    """

    PACKRAT_ARGS = ['uv_scale', 'uv_shape', 'uv_los', 'uv_area']

    def __init__(self):
        """Constructor for a NullFOV.

        The U-axis is assumed to align with X and the V-axis aligns with Y.

        Input:
            uv_scale    a single value, tuple or Pair defining the ratios dx/du
                        and dy/dv. For example, if (u,v) are in units of
                        arcseconds, then
                            uv_scale = Pair((pi/180/3600.,pi/180/3600.))
                        Use the sign of the second element to define the
                        direction of increasing V: negative for up, positive for
                        down.

            uv_shape    a single value, tuple or Pair defining size of the field
                        of view in pixels. This number can be non-integral if
                        the detector is not composed of a rectangular array of
                        pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the nominal line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.

            uv_area     an optional parameter defining the nominal field of view
                        of a pixel. If not provided, the area is calculated
                        based on the area of the central pixel.
        """

        self.uv_los = Pair.ZEROS
        self.uv_scale = Pair.ONES
        self.uv_shape = (1,1)
        self.uv_area = 1.

    def uv_from_xy(self, xy_pair, derivs=False):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.
        """

        return Pair.ZEROS

    def xy_from_uv(self, uv_pair, derivs=False):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.
        """

        return Pair.ZEROS

    # Overrides of the default FOV functions

    def area_factor(self, uv_pair, **keywords):
        """The relative area of a pixel or other sensor at (u,v).

        Results are scaled to the nominal pixel area.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        return Scalar.ONE

    def los_from_xy(self, xy_pair, derivs=False):
        """Return the unit line-of-sight vector for camera coordinates (x,y).

        Note that this is vector points in the direction _opposite_ to the path
        of arriving photons.

        If derivs is True, then derivatives in (x,y) get propagated forward
        into the components of the line-of-sight vector.
        """

        return Vector3.ZAXIS

    def xy_from_los(self, los, derivs=False):
        """Return camera frame coordinates (x,y) given a line of sight.

        Lines of sight point outward from the camera, near the Z-axis, and are
        therefore opposite to the direction in which a photon is moving. The
        length of the vector is ignored.

        If derivs is True, then derivatives in the components of the line of
        sight get propagated forward into the components of the (x,y)
        coordinates.
        """

        return Pair.ONES

    def los_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return the line of sight vector given FOV coordinates (u,v).

        The los points  the direction specified by coordinate Pair (u,v). Note
        that this is the direction _opposite_ to that of the arriving photon.

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        return Vector3.ZAXIS

    def uv_from_los(self, los, derivs=False, **keywords):
        """Return FOV coordinates (u,v) given a line of sight vector.

        The los points  the direction specified by coordinate Pair (u,v). Note
        that this is the direction _opposite_ to that of the arriving photon.

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        return Pair.ONES

    def uv_is_outside(self, uv_pair, inclusive=True, uv_min=None, uv_max=None):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            inclusive   True to interpret coordinate values at the upper end of
                        each range as inside the FOV; False to interpet them as
                        outside.
            uv_min      an integer Pair representing the lower (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.
            uv_max      an integer Pair representing the upper (u,v) corner of
                        the area observed at the FOV's active area; None for the
                        full FOV.

        Return:         a boolean indicating True where the point is outside the
                        FOV.
        """

        return True

    def u_or_v_is_outside(self, uv_coord, uv_index, inclusive=True,
                                          uv_min=None, uv_max=None):
        """Return a boolean mask identifying coordinates outside the FOV.

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

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """

        return True

    def xy_is_outside(self, xy_pair, inclusive=True, uv_min=None, uv_max=None,
                                                                    **keywords):
        """Return a boolean mask identifying coordinates outside the FOV.

        Input:
            xy_pair     a Pair of (x,y) coordinates, assuming z == 1.
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

        return True

    def los_is_outside(self, los, inclusive=True, uv_min=None, uv_max=None,
                                                               **keywords):
        """Return a boolean mask identifying lines of sight outside the FOV.

        Input:
            los         an outward line-of-sight vector.
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

        return True

    def nearest_uv(self, uv_pair, remask=False):
        """Return the closest (u,v) coordinates inside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            remask      True to mask the points outside the boundary.

        Return:         a new Pair of (u,v) coordinates.
        """

        return Pair.ZEROS

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_NullFOV(unittest.TestCase):

    def runTest(self):

        # No tests here - TBD

        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
