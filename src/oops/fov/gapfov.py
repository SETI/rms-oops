##########################################################################################
# oops/fov/gapfov.py
##########################################################################################

import numbers
import numpy as np

from polymath import Pair
from oops.fov import FOV


class GapFOV(FOV):
    """A subclass of FOV in which there are gaps between the individual pixels."""

    def __init__(self, fov, uv_size):
        """Constructor for a GapFOV.

        Pixels in the new FOV have the same origins as in the given FOV, but their `(u,v)`
        extent is reduced.

        Parameters:
            fov (FOV): Object relative to which this GapFOV is defined.
            uv_size (float, tuple, or Pair): The sizes of the new pixels relative to the
                sizes of the originals.
        """

        self.fov = fov

        # Allow for one or two inputs
        if isinstance(uv_size, numbers.Real):
            uv_size = (uv_size, uv_size)

        # Convert to Pair
        self.uv_size = Pair.as_pair(uv_size)
        self.uv_size_inv = Pair.as_pair((1./self.uv_size.vals[0],
                                         1./self.uv_size.vals[1]))

        self._uv_size2 = self.uv_size.vals[0] * self.uv_size.vals[1]
        self._uv_shape_tuple = tuple(fov.uv_shape.vals)

        # Required fields; the pixel grid is unchanged, so only the pixel size shrinks
        self.uv_scale = self.fov.uv_scale.element_mul(self.uv_size)
        self.uv_los   = self.fov.uv_los
        self.uv_area  = self.fov.uv_area * self._uv_size2
        self.uv_shape = self.fov.uv_shape

    def __getstate__(self):
        self.refresh()
        return (self.fov, self.uv_size)

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
            Pair: The transformed `(x,y)` coordinates in the camera's frame.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        uv_int = uv_pair.int(top=self._uv_shape_tuple)
        uv_frac = uv_pair - uv_int
        uv = uv_int + uv_frac.element_mul(self.uv_size)

        return self.fov.xy_from_uvt(uv, time=time, derivs=derivs, remask=remask,
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
        uv_int = uv_pair.int(top=self._uv_shape_tuple)
        uv_frac = (uv_pair - uv_int).element_mul(self.uv_size_inv)

        # Clip (u,v) in the gaps
        for k in range(2):
            in_gap = ((uv_frac.vals[...,k] > 1.) &
                      (uv_int.vals[...,k] < self.uv_shape.vals[k]))
            uv_frac.vals[...,k] = np.where(in_gap, 1., uv_frac.vals[...,k])
            if remask:
                uv_frac = uv_frac.remask_or(in_gap)

        return uv_int + uv_frac

##########################################################################################
