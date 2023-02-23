################################################################################
# oops/fov/nullfov.py: NullFOV subclass of class FOV
################################################################################

from polymath import Boolean, Scalar, Pair, Vector3

from . import FOV

class NullFOV(FOV):
    """A subclass of FOV that describes an instrument with no field of view,
    e.g., an in situ instrument.
    """

    #===========================================================================
    def __init__(self):
        """Constructor for a NullFOV."""

        self.uv_los = Pair.ZEROS
        self.uv_scale = Pair.ONES
        self.uv_shape = (1,1)
        self.uv_area = 1.

    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        self.__init__()

    #===========================================================================
    def xy_from_uvt(self, uv_pair, time=None, derivs=False, remask=False):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute times. Ignored by NullFOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        return Pair.ZEROS

    #===========================================================================
    def uv_from_xyt(self, xy_pair, time=None, derivs=False, remask=False):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute times. Ignored by NullFOV.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        return Pair.ZEROS

    ############################################################################
    # Overrides of the default FOV functions
    ############################################################################

    def area_factor(self, uv_pair, time=None, remask=False):
        """The relative area of a pixel or other sensor at (u,v).

        Results are scaled to the nominal pixel area.

        Input:
            uv_pair     Pair of (u,v) coordinates.
            time        Scalar of optional absolute times. Ignored by NullFOV.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         relative area of the pixel at (u,v), as a Scalar.
        """

        return Scalar.ONE

    #===========================================================================
    def los_from_xy(self, xy_pair, derivs=False):
        """The unit line-of-sight vector for camera coordinates (x,y).

        Note that this vector points in the direction _opposite_ to the path
        of arriving photons.

        Input:
            xy_pair     Pairs of (x,y) coordinates.
            derivs      True to propagate any derivatives in (x,y) forward
                        into the line-of-sight vector.

        Return:         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
        """

        return Vector3.ZAXIS

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

        return Pair.ZEROS

    #===========================================================================
    def los_from_uvt(self, uv_pair, time=None, derivs=False, remask=False):
        """The line of sight vector given FOV coordinates (u,v) at the specified
        time.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Input:
            uv_pair     Pair of (u,v) coordinates.
            time        Scalar of optional absolute times. Ignored by NullFOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Vector3 direction of the line of sight in the camera's
                        frame.
        """

        return Vector3.ZAXIS

    #===========================================================================
    def uv_from_los_t(self, los, time=None, derivs=False, remask=False):
        """The FOV coordinates (u,v) given a line of sight vector at the
        specified time.

        The los points in the direction specified by coordinate Pair (u,v).
        Note that this is the direction _opposite_ to that of the arriving
        photon.

        Input:
            los         Vector3 direction of the line of sight in the camera's
                        coordinate frame.
            time        Scalar of optional absolute times. Ignored by NullFOV.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned line of sight.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.

        Return:         Pair of (u,v) coordinates in the FOV.
        """

        return Pair.ONES

    #===========================================================================
    def uv_is_outside(self, uv_pair, time=None, inclusive=True,
                            uv_min=None, uv_max=None):
        """A Boolean mask identifying coordinates outside the FOV.

        Input:
            uv_pair     a Pair of (u,v) coordinates.
            time        Scalar of optional absolute times. Ignored by NullFOV.
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

        # A shapeless return value should be OK
        return Boolean.TRUE

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

        Return:         a boolean NumPy array indicating True where the point is
                        outside the FOV.
        """

        # A shapeless return value should be OK
        return Boolean.TRUE

    #===========================================================================
    def xy_is_outside(self, xy_pair, time=None, inclusive=True,
                            uv_min=None, uv_max=None):
        """A boolean mask identifying coordinates outside the FOV.

        Input:
            xy_pair     a Pair of (x,y) coordinates, assuming z == 1.
            time        Scalar of optional absolute times. Ignored by NullFOV.
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

        # A shapeless return value should be OK
        return Boolean.TRUE

    #===========================================================================
    def los_is_outside(self, los, time=None, inclusive=True,
                             uv_min=None, uv_max=None):
        """A boolean mask identifying lines of sight outside the FOV.

        Input:
            los         an outward line-of-sight vector.
            time        Scalar of optional absolute times. Ignored by NullFOV.
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

        # A shapeless return value should be OK
        return Boolean.TRUE

    #===========================================================================
    def nearest_uv(self, uv_pair, remask=False):
        """The closest (u,v) coordinates inside the FOV.

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
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
