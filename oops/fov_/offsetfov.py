################################################################################
# oops/fov_/offsetfov.py: Offset subclass of FOV
################################################################################

import numpy as np

from polymath import *

from oops.fov_.fov  import FOV
from oops.fittable import Fittable

class OffsetFOV(FOV, Fittable):
    """A FOV object in which the line of sight has been shifted relative to
    another FOV. This is typically used for image navigation and pointing
    corrections.
    """

    PACKRAT_ARGS = ['fov', 'uv_offset', 'xy_offset']

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
        assert (uv_offset is None) or (xy_offset is None)

        self.uv_offset = uv_offset
        self.xy_offset = xy_offset

        if self.uv_offset is not None:
            self.xy_offset = self.fov.xy_from_uv(self.uv_offset +
                                                 self.fov.uv_los)

        elif self.xy_offset is not None:
                self.uv_offset = (self.fov.uv_from_xy(self.xy_offset) -
                                  self.fov.uv_los)

        else:                                   # default is a (0,0) offset
                self.uv_offset = Pair.ZERO
                self.xy_offset = Pair.ZERO

        # Required attributes of an FOV
        self.uv_shape = self.fov.uv_shape
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area
        self.uv_los = self.fov.uv_los - self.uv_offset

        # Required attributes for Fittable
        self.nparams = 2
        self.param_name = 'uv_offset'
        self.cache = {}     # not used

    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        """Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        uv_pair = Pair.as_pair(uv_pair, derivs)
        old_xy = self.fov.xy_from_uv(uv_pair, derivs, **keywords)
        return old_xy - self.xy_offset

    def uv_from_xy(self, xy_pair, derivs=False, **keywords):
        """Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """

        xy_pair = Pair.as_pair(xy_pair, derivs)
        return self.fov.uv_from_xy(xy_pair + self.xy_offset, derivs,
                                   **keywords)

    ########################################
    # Fittable interface
    ########################################

    def set_params(self, params):
        """Redefine the Fittable object, using this set of parameters.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        self.uv_offset = Pair.as_pair(params)
        self.xy_offset = self.fov.xy_from_uv(self.uv_offset - self.fov.uv_los)

    def copy(self):
        """Return a deep copy of the Fittable object.

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
