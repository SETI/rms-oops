################################################################################
# oops/fov/offsetfov.py: OffsetFOV subclass of FOV
################################################################################

from polymath      import Pair
from oops.fittable import Fittable
from oops.fov      import FOV

class OffsetFOV(FOV, Fittable):
    """FOV subclass in which the line of sight has been shifted relative to
    another FOV. This is typically used for image navigation and pointing
    corrections.
    """

    #===========================================================================
    def __init__(self, fov, uv_offset=None, xy_offset=None):
        """Constructor for an OffsetFOV.

        Inputs:
            fov         the reference FOV from which this FOV has been offset.

            uv_offset   a tuple or Pair defining the offset of the new FOV
                        relative to the old. This can be understood as having
                        the effect of shifting predicted image geometry relative
                        to what the image actually shows.

            xy_offset   an alternative input, in which the offset is given in
                        (x,y) coordinates rather than (u,v) coordinates.

        Note that the Fittable interface uses the uv_offset, not the alternative
        xy_offset input parameters.
        """

        self.fov = fov

        # Deal with alternative inputs:
        if (uv_offset is not None) and (xy_offset is not None):
            raise ValueError('only one of uv_offset and xy_offset can be '
                             + 'specified')

        self.uv_offset = uv_offset
        self.xy_offset = xy_offset

        if self.uv_offset is not None:
            self.xy_offset = self.fov.xy_from_uv(self.uv_offset +
                                                 self.fov.uv_los)

        elif self.xy_offset is not None:
            self.uv_offset = (self.fov.uv_from_xy(self.xy_offset) -
                              self.fov.uv_los)

        else:                                   # default is a (0,0) offset
            self.uv_offset = Pair.ZEROS
            self.xy_offset = Pair.ZEROS

        # Required attributes of an FOV
        self.uv_shape = self.fov.uv_shape
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area
        self.uv_los = self.fov.uv_los - self.uv_offset

        # Required attributes for Fittable
        self.nparams = 2
        self.param_name = 'uv_offset'
        self.cache = {}     # not used

    def __getstate__(self):
        return (self.fov, self.uv_offset, self.xy_offset)

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

        uv_pair = Pair.as_pair(uv_pair, derivs)
        old_xy = self.fov.xy_from_uvt(uv_pair, time=time, derivs=derivs,
                                      remask=remask, **keywords)
        return old_xy - self.xy_offset

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

        xy_pair = Pair.as_pair(xy_pair, derivs)
        return self.fov.uv_from_xyt(xy_pair + self.xy_offset, time=time,
                                    derivs=derivs, remask=remask, **keywords)

    ############################################################################
    # Fittable interface
    ############################################################################

    def set_params(self, params):
        """Redefine the Fittable object, using this set of parameters.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        self.uv_offset = Pair.as_pair(params)
        self.xy_offset = self.fov.xy_from_uv(self.uv_offset - self.fov.uv_los)

    #===========================================================================
    def copy(self):
        """A deep copy of the Fittable object.

        The copy can be safely modified without affecting the original.
        """

        return OffsetFOV(self.fov, self.uv_offset.copy(), xy_offset=None)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_OffsetFOV(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
