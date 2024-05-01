################################################################################
# oops/fov/subsampledfov.py: SubsampledFOV subclass of FOV
################################################################################

from polymath import Pair
from oops.fov import FOV

class SubsampledFOV(FOV):
    """Subclass of FOV in which the pixels of a given base FOV class are
    re-scaled.
    """

    #===========================================================================
    def __init__(self, fov, rescale):
        """Constructor for a SubsampledFOV.

        Returns a new FOV object in which the pixel size has been modified.
        The origin and the optic axis are unchanged.

        Inputs:
            fov         the FOV object within which this subsampledFOV is
                        defined.

            rescale     a single value, tuple or Pair defining the sizes of the
                        new pixels relative to the sizes of the originals.
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
        return (self.fov, self.rescale)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def xy_from_uvt(self, uv_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (x,y) camera frame coordinates given the FOV coordinates (u,v) at
        the specified time.

        Input:
            uv_pair     (u,v) coordinate Pair in the FOV.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (u,v) get propagated into
                        the returned (x,y) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as uv_pair, giving the transformed
                        (x,y) coordinates in the camera's frame.
        """

        uv_pair = Pair.as_pair(uv_pair, recursive=derivs)
        return self.fov.xy_from_uvt(self.rescale.element_mul(uv_pair),
                                    time=time, derivs=derivs, remask=remask,
                                    **keywords)

    #===========================================================================
    def uv_from_xyt(self, xy_pair, time=None, derivs=False, remask=False,
                                                            **keywords):
        """The (u,v) FOV coordinates given the (x,y) camera frame coordinates at
        the specified time.

        Input:
            xy_pair     (x,y) Pair in FOV coordinates.
            time        Scalar of optional absolute time in seconds.
            derivs      If True, any derivatives in (x,y) get propagated into
                        the returned (u,v) Pair.
            remask      True to mask (u,v) coordinates outside the field of
                        view; False to leave them unmasked.
            **keywords  Additional keywords arguments are passed directly to the
                        reference FOV.

        Return:         Pair of same shape as xy_pair, giving the computed (u,v)
                        FOV coordinates.
        """

        xy_pair = Pair.as_pair(xy_pair, recursive=derivs)
        uv_pair = self.fov.uv_from_xyt(xy_pair, time=time, derivs=derivs,
                                                remask=remask, **keywords)
        uv_new = uv_pair.element_div(self.rescale)

        return uv_new

################################################################################
