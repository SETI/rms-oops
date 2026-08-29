##########################################################################################
# oops/fov/platescale.py: Platescale subclass of class FOV
##########################################################################################

from oops.fittable import Fittable
from oops.fov      import FOV
from polymath      import Pair


class Platescale(FOV, Fittable):
    """An FOV defined by applying a plate scale to another FOV."""

    def __init__(self, factor, /, fov):
        """Constructor for a Platescale FOV.

        Parameters:
            factor (float): The scale factor to apply to the given FOV. A value greater
                than one enlarges the FOV.
            fov (FOV): The FOV object to which the scale factor applies.
        """

        self.factor = factor
        self.fov = fov

        self.uv_los = self.fov.uv_los
        self.uv_shape = self.fov.uv_shape

        self._refresh()

    ######################################################################################
    # Fittable API
    ######################################################################################

    nparams = 1

    def _refresh(self):
        self.uv_scale = self.fov.uv_scale * self.factor

        # (x,y) are scaled by the factor, so a unit step in (u,v) covers the square of
        # the factor in area; area_factor() divides by this value
        self.uv_area = self.fov.uv_area * self.factor**2

    def _set_params(self, params):
        """Redefine the scale factor of this Platescale FOV."""

        self.factor = params[0]

    @property
    def params(self):
        """The fitted parameters, the scale factor as a tuple of one float."""

        return (self.factor,)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self.factor, self.fov)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    ######################################################################################
    # FOV API
    ######################################################################################

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given FOV coordinates `(u,v)`.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in `(u,v)` get propagated
                into the returned `(x,y)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: `(x,y)` coordinates in the FOV's frame.
        """

        xy_pair = self.fov.xy_from_uvt(uv_pair, time=time, derivs=derivs, remask=remask,
                                       **kwargs)
        return xy_pair * self.factor

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given `(x,y)` camera frame coordinates.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB.
            derivs (bool, optional): If True, any derivatives in `(x,y)` get propagated
                into the returned `(u,v)` coordinates.
            remask (bool, optional): True to mask `(u,v)` coordinates outside the field of
                view; False to leave them unmasked.
            **kwargs: Additional parameters that might affect the transform can be
                included as keyword arguments.

        Returns:
            Pair: `(u,v)` pixel coordinates in the FOV.
        """

        xy_pair = Pair.as_pair(xy_pair)
        return self.fov.uv_from_xyt(xy_pair / self.factor, time=time, derivs=derivs,
                                    remask=remask, **kwargs)

##########################################################################################
