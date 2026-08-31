##########################################################################################
# oops/fov/subarray.py
##########################################################################################

from polymath import Pair
from oops.fov import FOV


class Subarray(FOV):
    """Subclass of FOV that describes a rectangular region of a larger FOV."""

    def __init__(self, fov, new_los, uv_shape, uv_los=None):
        """Constructor for a Subarray.

        In the returned FOV object, the ICS origin and/or the optic axis have been
        modified.

        Parameters:
            fov (FOV): Object within which this subarray is defined.
            new_los (tuple or Pair): The location of the subarray's line of sight in the
                `(u,v)` coordinates of the original FOV.
            uv_shape (float, tuple, or Pair): The new size of the field of view in pixels.
            uv_los (float, tuple, or Pair, optional): The `(u,v)` coordinates of the new
                line of sight. By default, this is the midpoint of the rectangle, i.e.,
                `uv_shape/2`.
        """

        self.fov = fov
        self.new_los_in_old_uv  = Pair.as_pair(new_los).as_float()
        self.new_los_wrt_old_xy = fov.xy_from_uv(self.new_los_in_old_uv)
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_readonly()

        self.new_origin_in_old_uv = self.new_los_in_old_uv - self.uv_los

        self.new_los_in_old_uv.as_readonly()
        self.new_los_wrt_old_xy.as_readonly()
        self.uv_shape.as_readonly()
        self.uv_los.as_readonly()
        self.new_origin_in_old_uv.as_readonly()

        # Required fields
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area

    def __getstate__(self):
        self.refresh()
        return (self.fov, self.new_los_in_old_uv, self.uv_shape, self.uv_los)

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

        old_xy = self.fov.xy_from_uvt(self.new_origin_in_old_uv + uv_pair,
                                      time=time, derivs=derivs, remask=remask,
                                      **kwargs)
        return old_xy - self.new_los_wrt_old_xy

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

        old_uv = self.fov.uv_from_xyt(self.new_los_wrt_old_xy + xy_pair,
                                      time=time, derivs=derivs, remask=remask,
                                      **kwargs)
        return old_uv - self.new_origin_in_old_uv

##########################################################################################
