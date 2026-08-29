##########################################################################################
# oops/fov/flatfov.py: FlatFOV subclass of class FOV
##########################################################################################

import numpy as np

from polymath import Pair
from oops.fov import FOV

class FlatFOV(FOV):
    """FOV subclass that describes a field of view that is free of distortion,
    implementing an exact pinhole ("gnomonic") camera model.
    """

    def __init__(self, uv_scale, uv_shape, *, uv_los=None, uv_area=None):
        """Constructor for a FlatFOV.

        The U-axis is assumed to align with X and the V-axis aligns with Y.

        Parameters:
            uv_scale (float, tuple, or Pair): The ratios `dx/du` and `dy/dv`. For
                example, if `(u,v)` are in units of arcseconds, then::

                    uv_scale = Pair((pi/180/3600.,pi/180/3600.))

                Use the sign of the second element to define the direction of increasing
                `v`: negative for up, positive for down.
            uv_shape (float, tuple, or Pair): The size of the field of view in pixels.
                This number can be non-integral if the detector is not composed of a
                rectangular array of pixels.
            uv_los (float, tuple, or Pair, optional): The `(u,v)` coordinates of the
                nominal line of sight. By default, this is the midpoint of the rectangle,
                i.e., `uv_shape/2`.
            uv_area (float, optional): The nominal area of a pixel in steradians. If not
                provided, it is derived from `uv_scale`.
        """

        self.uv_scale = Pair.as_pair(uv_scale).as_float().as_readonly()
        self.uv_shape = Pair.as_pair(uv_shape).as_readonly()

        if uv_los is None:
            self.uv_los = self.uv_shape / 2.
        else:
            self.uv_los = Pair.as_pair(uv_los).as_float().as_readonly()

        if uv_area is None:
            self.uv_area = np.abs(self.uv_scale.vals[0] * self.uv_scale.vals[1])
        else:
            self.uv_area = uv_area

        scale = Pair.as_pair(uv_scale).as_readonly()

        self.dxy_duv = Pair([[  scale.vals[0], 0.],
                             [0.,   scale.vals[1]]], drank=1).as_readonly()
        self.duv_dxy = Pair([[1/scale.vals[0], 0.],
                             [0., 1/scale.vals[1]]], drank=1).as_readonly()

    def __getstate__(self):
        self.refresh()
        return (self.uv_scale, self.uv_shape, self.uv_los, self.uv_area)

    def __setstate__(self, state):
        (uv_scale, uv_shape, uv_los, uv_area) = state
        self.__init__(uv_scale, uv_shape, uv_los=uv_los, uv_area=uv_area)
        self.freeze()

    def xy_from_uvt(self, uv_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(x,y)` camera frame coordinates given the FOV coordinates `(u,v)` at the
        specified time.

        Parameters:
            uv_pair (Pair): `(u,v)` coordinates in this FOV.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by FlatFOV.
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
        if remask:
            uv_pair = uv_pair.remask_or(self.uv_is_outside(uv_pair).vals)

        return (uv_pair - self.uv_los).element_mul(self.uv_scale)

    def uv_from_xyt(self, xy_pair, time=None, *, derivs=False, remask=False, **kwargs):
        """The `(u,v)` FOV coordinates given `(x,y)` camera frame coordinates.

        Parameters:
            xy_pair (Pair): `(x,y)` coordinates in this FOV, assuming `z = 1`.
            time (Scalar, optional): Absolute time in seconds TDB. Ignored by FlatFOV.
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
        uv_pair = xy_pair.element_div(self.uv_scale) + self.uv_los
        if remask:
            uv_pair = uv_pair.remask_or(self.uv_is_outside(uv_pair).vals)

        return uv_pair

##########################################################################################
