################################################################################
# oops/fov_/offset.py: Offset subclass of FOV
#
# 3/21/12 MRS - New.
# 10/28/12 MRS - Complete update to accommodate the Fittable interface.
# 1/28/13 MRS - The previous version was found to be mathematically incorrect,
#   in that offsets should be applied to the un-distorted (x,y) coordinates
#   rather than to the (possibly distorted) (u,v) coordinates. The code was
#   revised to account for this, with the original behavior preserved but
#   deprecated. The deprecated behavior is still the default, but this may be
#   revised in a future release.
#   The option to specify an offset as an (x,y) offset rather than a (u,v)
#   offset was also added.
################################################################################

import numpy as np
import warnings

from oops.fov_.fov import FOV
from oops.fittable import Fittable
from oops.array_   import *

class Offset(FOV, Fittable):

    def __init__(self, fov, uv_offset=None, xy_offset=None,
                 deprecated_behavior=True, deprecation_warning=True):
        """Returns a new FOV object in which the line of sight has been shifted
        relative to another FOV. This is typically used for image navigation and
        pointing corrections.

        Inputs:
            fov         the FOV object from which this FOV has been displaced.

            uv_offset   a tuple or Pair defining the offset of the new FOV
                        relative to the old. This can be understood as having
                        the effect of shifting predicted image geometry relative
                        to what the image actually shows.

            xy_offset   an alternative input, in which the offset is given in
                        (x,y) coordinates rather than (u,v) coordinates.

            deprecated_behavior     True to use deprecated behavior, in which
                                    the offset is applied to the (possibly
                                    distorted) pixel grid. False to use the
                                    recommended behavior, in which the offset is
                                    applied as a constant (x,y) shift.

            deprecation_warning     True to warn the user when deprecated
                                    behavior is being used.

        Note that the Fittable interface uses the uv_offset, not the alternative
        xy_offset input parameters.
        """

        self.fov = fov

        # Identify deprecated behavior
        self.deprecated_behavior = deprecated_behavior
        if (deprecated_behavior and deprecation_warning):
            warnings.warn("deprecated use of fov.Offset; " +
                          "deprecated_behavior=False is recommended, " +
                          "or else set deprecation_warning=False to suppress " +
                          "this warning", DeprecationWarning)

        # Deal with alternative inputs:
        assert (uv_offset is None) or (xy_offset is None)

        self.uv_offset = uv_offset
        self.xy_offset = xy_offset

        if self.uv_offset is not None:
            self.xy_offset = self.fov.xy_from_uv(self.uv_offset -
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

        # Required attributes for Fittable
        self.nparams = 2
        self.cache = {}     # not used

    def xy_from_uv(self, uv_pair, extras=(), derivs=False):
        """Returns a Pair of (x,y) spatial coordinates in units of radians,
        given a Pair of coordinates (u,v).

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_duv", which
        contains the partial derivatives d(x,y)/d(u,v) as a MatrixN with item
        shape [2,2].
        """

        if self.deprecated_behavior:
            new_xy = self.fov.xy_from_uv(uv_pair - self.uv_offset, extras,
                                                                   derivs)

            if derivs:
                old_xy = self.fov.xy_from_uv(uv_pair, extras, derivs)
                new_xy.insert_subfield("d_uv", old_xy.d_duv)

        else:
            old_xy = self.fov.xy_from_uv(uv_pair, extras, derivs)
            new_xy = old_xy - self.xy_offset

            if derivs:
                new_xy.insert_subfield("d_uv", old_xy.d_duv)

        return new_xy

    def uv_from_xy(self, xy_pair, extras=(), derivs=False):
        """Returns a Pair of coordinates (u,v) given a Pair (x,y) of spatial
        coordinates in radians.

        Additional parameters that might affect the transform can be included
        in the extras argument.

        If derivs is True, then the returned Pair has a subarrray "d_dxy", which
        contains the partial derivatives d(u,v)/d(x,y) as a MatrixN with item
        shape [2,2].
        """

        if self.deprecated_behavior:
            old_uv = self.fov.uv_from_xy(xy_pair, extras, derivs)
            new_uv = old_uv + self.uv_offset

            if derivs:
                new_uv.insert_subfield("d_dxy", old_uv.d_duv)

        else:
            new_uv = self.fov.uv_from_xy(xy_pair + self.xy_offset, extras,
                                                                   derivs)

        return new_uv

    ########################################
    # Fittable interface
    ########################################

    def set_params(self, params):
        """Redefines the Fittable object, using this set of parameters. Unlike
        method set_params(), this method does not check the cache first.
        Override this method if the subclass should use a cache.

        Input:
            params      a list, tuple or 1-D Numpy array of floating-point
                        numbers, defining the parameters to be used in the
                        object returned.
        """

        self.uv_offset = Pair(params)

        if not self.deprecated_behavior:        # if self.xy_offset is needed...
            self.xy_offset = self.fov.xy_from_uv(self.uv_offset -
                                                 self.fov.uv_los)

    def get_params(self):
        """Returns the current set of parameters defining this fittable object.

        Return:         a Numpy 1-D array of floating-point numbers containing
                        the parameter values defining this object.
        """

        return self.uv_offset.vals

    def copy(self):
        """Returns a deep copy of the given object. The copy can be safely
        modified without affecting the original."""

        return Offset(self.fov, self.uv_offset.copy(),
                      deprecated_behavior = self.deprecated_behavior,
                      deprecation_warning = False)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Offset(unittest.TestCase):

    def runTest(self):

        # TBD
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
