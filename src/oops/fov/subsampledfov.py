##########################################################################################
# oops/fov/subsampledfov.py
##########################################################################################

from polymath import Pair
from oops.fov import FOV


class SubsampledFOV(FOV):
    """Subclass of FOV in which the pixels of a given FOV are re-scaled."""

    def __init__(self, fov, rescale):
        """Constructor for a SubsampledFOV.

        In the new FOV object, the pixel size has been modified. The origin and the optic
        axis are unchanged.

        Parameters:
            fov (FOV): Object within which this SubsampledFOV is defined.
            rescale (float, tuple, or Pair): The sizes of the new pixels relative to the
                sizes of the originals.
        """

        self.fov = fov
        self.rescale  = Pair.as_pair(rescale).as_readonly()
        self.rescale2 = self.rescale.vals[0] * self.rescale.vals[1]

        # Required fields
        self.uv_scale = self.fov.uv_scale.element_mul(self.rescale)
        self.uv_los   = self.fov.uv_los.element_div(self.rescale)
        self.uv_area  = self.fov.uv_area * self.rescale2

        self.uv_shape = self.fov.uv_shape.element_div(self.rescale).as_int()

    def __getstate__(self):
        self.refresh()
        return (self.fov, self.rescale)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

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
            Pair: The transformed `(x,y)` coordinates in the camera's frame, with the same
                shape as uv_pair.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        return self.fov.xy_from_uvt(self.rescale.element_mul(uv_pair),
                                    time=time, derivs=derivs, remask=remask,
                                    **kwargs)

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given the `(x,y)` camera frame coordinates at the
        specified time.

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
            Pair: The computed `(u,v)` FOV coordinates, with the same shape as xy_pair.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_pair = self.fov.uv_from_xyt(xy_pair, time=time, derivs=derivs,
                                                remask=remask, **kwargs)
        uv_new = uv_pair.element_div(self.rescale)

        return uv_new

##########################################################################################
