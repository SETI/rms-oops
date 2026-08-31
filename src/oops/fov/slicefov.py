##########################################################################################
# oops/fov/slicefov.py
##########################################################################################

from polymath import Pair
from oops.fov import FOV


class SliceFOV(FOV):
    """A subclass of FOV in which only a slice of another FOV's (u,v) array is used, but
    the geometry is unchanged.

    This differs from a Subarray in that the optic axis is not modified.
    """

    def __init__(self, fov, origin, shape):
        """Constructor for a SliceFOV.

        Parameters:
            fov (FOV): The reference FOV object within which this slice is defined.
            origin (tuple or Pair): The location of this slice's pixel `(0,0)` in the
                coordinates of the reference FOV.
            shape (float, tuple, or Pair): The new shape of the field of view in pixels.
        """

        self.fov = fov
        self.uv_origin = Pair.as_pair(origin).as_int().as_readonly()
        self.uv_shape  = Pair.as_pair(shape).as_int().as_readonly()

        # Required fields
        self.uv_los   = self.fov.uv_los - self.uv_origin
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        self.refresh()
        return (self.fov, self.uv_origin, self.uv_shape)

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

        return self.fov.xy_from_uvt(uv_pair + self.uv_origin, time=time,
                                    derivs=derivs, remask=remask, **kwargs)

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

        new_uv = self.fov.uv_from_xyt(xy_pair, time=time, derivs=derivs,
                                               remask=remask, **kwargs)
        return new_uv - self.uv_origin

##########################################################################################
