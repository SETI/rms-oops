##########################################################################################
# oops/fov/platescale.py: Platescale subclass of class FOV
##########################################################################################

from oops.fittable import Fittable
from oops.fov      import FOV
from polymath      import Pair


class Platescale(FOV, Fittable):
    """An FOV defined by applying a plate scale to another FOV.

    PLACEHOLDER CODE. "CONCEPTUALLY" CORRECT BUT NOT YET TESTED.
    """

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
        self.uv_area = self.fov.uv_area

        self._refresh()

    ######################################################################################
    # Fittable API
    ######################################################################################

    def _refresh(self):
        self.uv_scale = self.fov.uv_scale * self.factor

    def _set_params(self, params):
        self.factor = params[0]

    @property
    def _params(self):
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

    def xy_from_uvt(self, uv_pair, time=None, derivs=False, remask=False, **keywords):
        """The (x,y) camera frame coordinates given FOV coordinates (u,v).

        Parameters:
            uv_pair (Pair or array-like): (u,v) pixel coordinates in the FOV.
            time (Scalar, array-like, or float, optional): Time in TDB seconds, ignored by
                time-independent FOVs.
            derivs (bool, optional): True to propagate any derivatives in (u,v) into the
                returned (x,y) Pair.
            remask (bool, optional): True to mask (u,v) coordinates that fall outside the
                field of view; False to leave them unmasked.
            **keywords: Additional keywords arguments are passed directly to the reference
                FOV.

        Returns:
            (Pair): (x,y) coordinates in the FOV's frame.
        """

        xy_pair = self.fov.xy_from_uvt(uv_pair, time=time, derivs=derivs, remask=remask,
                                       **keywords)
        return xy_pair * self.factor

    def uv_from_xyt(self, xy_pair, time=None, derivs=False, remask=False, **keywords):
        """The (u,v) FOV coordinates given (x,y) camera frame coordinates.

        Parameters:
            xy_pair (Pair or array-like): (x,y) coordinates in the FOV.
            time (Scalar, array-like, or float, optional): Time in TDB seconds, ignored by
                time-independent FOVs.
            derivs (bool, optional): True to propagate any derivatives in (x,y) into the
                returned (u,v) Pair.
            remask (bool, optional): True to mask (u,v) coordinates that fall outside the
                field of view; False to leave them unmasked.
            **keywords: Additional keywords arguments are passed directly to the reference
                FOV.

        Returns:
            (Pair): (u,v) pixel coordinates in the FOV.
        """

        xy_pair = Pair.as_pair(xy_pair)
        return self.fov.uv_from_xyt(xy_pair / self.factor, time=time, derivs=derivs,
                                    remask=remask, **keywords)

##########################################################################################
