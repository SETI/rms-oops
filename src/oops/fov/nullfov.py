##########################################################################################
# oops/fov/nullfov.py: NullFOV subclass of class FOV
##########################################################################################

from polymath import Boolean, Scalar, Pair, Vector3
from oops.fov import FOV

class NullFOV(FOV):
    """A subclass of FOV that describes an instrument with no field of view, e.g., an in
    situ instrument.
    """

    def __init__(self):
        """Constructor for a NullFOV."""

        self.uv_los = Pair.ZEROS
        self.uv_scale = Pair.ONES
        self.uv_shape = Pair((1,1))
        self.uv_area = 1.

    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        self.__init__()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The transformed `(x,y)` coordinates in the camera's frame, with the same
                shape as uv_pair.
        """

        return Pair.ZEROS

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: The computed `(u,v)` FOV coordinates, with the same shape as xy_pair.
        """

        return Pair.ZEROS

    ######################################################################################
    # Overrides of the default FOV functions
    ######################################################################################

    def area_factor(self, uv_pair, time=None, *, remask=False, **kwargs):
        """The relative area of a pixel or other sensor at `(u,v)`.

        Results are scaled to the nominal pixel area.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Scalar: Relative area of the pixel at `(u,v)`.
        """

        return Scalar.ONE

    def los_from_xy(self, xy_pair, *, derivs=False):
        """The unit line-of-sight vector for camera coordinates `(x,y)`.

        Note that this vector points in the direction _opposite_ to the path of arriving
        photons.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            derivs (bool, optional): True to propagate any derivatives of `(x,y)` into the
                returned line of sight.

        Returns:
            Vector3: Direction of the line of sight in the camera's coordinate frame.
        """

        return Vector3.ZAXIS

    def xy_from_los(self, los, *, derivs=False):
        """Camera frame coordinates `(x,y)` given a line of sight.

        Lines of sight point outward from the camera, near the Z-axis, and are therefore
        opposite to the direction in which a photon is moving. The length of the vector is
        ignored.

        Parameters:
            los (Vector3): Direction of the line of sight in the FOV's frame.
            derivs (bool, optional): True to propagate any derivatives of `los` into the
                returned coordinates.

        Returns:
            Pair: `(x,y)` coordinates in the camera's frame.
        """

        return Pair.ZEROS

    def los_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The line of sight vector given FOV coordinates `(u,v)` at the specified time.

        The los points in the direction specified by coordinate Pair `(u,v)`. Note that
        this is the direction _opposite_ to that of the arriving photon.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            derivs (bool, optional): True to propagate any derivatives of `(u,v)` into the
                returned line of sight.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Vector3: Direction of the line of sight in the camera's frame.
        """

        return Vector3.ZAXIS

    def uv_from_los_t(self, los, time=None, *, derivs=False, remask=False, **kwargs):
        """The FOV coordinates `(u,v)` given a line of sight vector at the specified time.

        The los points in the direction specified by coordinate Pair `(u,v)`. Note that
        this is the direction _opposite_ to that of the arriving photon.

        Parameters:
            los (Vector3): Direction of the line of sight in the FOV's frame.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            derivs (bool, optional): True to propagate any derivatives of `los` into the
                returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: `(u,v)` coordinates in the FOV.
        """

        return Pair.ZEROS

    def uv_is_outside(self, uv_pair, time=None, *, uv_min=None, uv_max=None,
                      inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.

        Returns:
            Boolean: True where the point is outside the FOV.
        """

        # A shapeless return value should be OK
        return Boolean.TRUE

    def u_or_v_is_outside(self, uv_pair, uv_index, *, uv_min=None, uv_max=None,
                          inclusive=True):
        """A Boolean mask identifying coordinates outside the FOV along one axis.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            uv_index (int): 0 to test u-coordinates; 1 to test v-coordinates.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV, of which only the element selected by `uv_index` is used; None for
                the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV, of which only the element selected by `uv_index` is used; None for
                the full FOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.

        Returns:
            Boolean: True where the point is outside the FOV along the specified axis.
        """

        # A shapeless return value should be OK
        return Boolean.TRUE

    def xy_is_outside(self, xy_pair, time=None, *, inclusive=True,
                      uv_min=None, uv_max=None, **kwargs):
        """A Boolean mask identifying coordinates outside the FOV.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Boolean: True where `xy_pair` is outside the FOV.
        """

        # A shapeless return value should be OK
        return Boolean.TRUE

    def los_is_outside(self, los, time=None, *, inclusive=True, uv_min=None, uv_max=None,
                       **kwargs):
        """A Boolean mask identifying lines of sight outside the FOV.

        Parameters:
            los (Vector3): Direction of the line of sight in the FOV's frame.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by NullFOV.
            inclusive (bool, optional): True to interpret coordinate values at the upper
                end of each range as inside the FOV; False to interpret them as outside.
            uv_min (Pair, optional): The lower `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            uv_max (Pair, optional): The upper `(u,v)` corner of the area observed in the
                FOV; None for the full FOV.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Boolean: True where `los` is outside the FOV.
        """

        # A shapeless return value should be OK
        return Boolean.TRUE

    def nearest_uv(self, uv_pair, *, remask=False):
        """The closest `(u,v)` coordinates inside the FOV.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.

        Returns:
            Pair: `(u,v)` coordinates.
        """

        return Pair.ZEROS

    def max_inversion_error(self, steps=30):
        """The largest error in pixels resulting from `(u,v) -> (x,y) -> (u,v)`.

        A NullFOV has no extent, so every `(u,v)` maps to the same line of sight and the
        transform is not invertible. Zero is returned rather than the size of the sampled
        region, which would be meaningless here.

        Parameters:
            steps (int, optional): The number of samples per axis. Ignored by NullFOV.

        Returns:
            float: Always zero.
        """

        return 0.

##########################################################################################
