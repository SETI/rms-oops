################################################################################
# oops/nav_/navigation.py: Class Navigation
################################################################################

import numpy as np

from polymath      import *
from oops.fittable import Fittable

class Navigation(Fittable):
    """The Navigation class defines various possible modifications of the
    geometry and timing of an observation. These include rotations, distortions
    and time-shifts.
    """

    def __init__(self, params):
        """Constructor for an abstract Navigation object. """

        pass

    ####################################

    def set_params(self, angles):
        """Part of the Fittable interface. Re-defines the Navigation given a
        new set of parameters."""

        pass

    ####################################

    def get_params(self):
        """Part of the Fittable interface. Returns the current parameters.
        """

        pass

    ####################################

    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        pass

    ####################################

    def distort(self, los, t, partials=False):
        """Applies the distortion to line-of-sight vectors. This should be
        applied to a vector in the instrument's coordinate frame, before it is
        converted to FOV (u,v) coordinates.

        Input:
            los         a Vector3 of line-of-sight vectors in the instrument's
                        coordinate frame.
            t           an optional Scalar defining the fractional time within
                        the observation at which the distortion applies.
            partials    if True, then the MatrixN of partial derivatives with
                        respect to the parameters are also returned as a
                        subfield "d_dnav" with item shape [3,N], where N is
                        the number of parameters.

        Return          a Vector3 of un-distorted vectors. If the input Vector3
                        has a subfield "d_dt", this is also distorted and
                        returned as a subfield.
        """

        # Default behavior is to do (almost) nothing
        los = Vector3.as_vector3(los)

        if partials:
            los.insert_subfield("d_dnav", self.dlos_dparams)

        return los

    ####################################

    def undistort(self, los, t, partials=False):
        """Removes the distortion from the line-of-sight vectors. This should
        be applied to a vector derived from FOV (u,v) coordinates, before
        conversion out of the instrument's coordinate frame.

        Input:
            los         a Vector3 of line-of-sight vectors based on FOV (u,v)
                        coordinates.
            t           an optional Scalar defining the fractional time within
                        the observation at which the distortion applies.
            partials    if True, then the MatrixN of partial derivatives with
                        respect to the parameters are also returned as a
                        subfield "d_dnav" with item shape [3,N], where N is
                        the number of parameters.

        Return          a Vector3 of un-distorted vectors. If the input Vector3
                        has a subfield "d_dt", this is also un-distorted and
                        returned as a subfield.
        """

        # Default behavior is to do (almost) nothing
        los = Vector3.as_vector3(los)

        if partials:
            los.insert_subfield("d_dnav", self.dlos_dparams)

        return los

    ####################################

    def shift(self, time, uv, partials=False):
        """Applies the time-shift to the observation. This should be applied to
        to times defined by external events, returning times applicable within
        the span of the observation.

        Input:
            time        a Scalar of external event times.
            uv          a Pair defining the (u,v) coordinates at which the
                        time-shift applies.
            partials    if True, then the partial derivatives with respect to
                        the parameters are also returned as a subfield "d_dnav".
                        The partials are represented by a MatrixN with item
                        shape [1,N], where N is the number of parameters.

        Return:         a Vector3 of times within the span of the observation.
        """

        # Default behavior is to do (almost) nothing
        time = Scalar.as_scalar(time)

        if partials:
            time.insert_subfield("d_dnav", self.dtime_dparams)

        return time

    ####################################

    def unshift(self, time, uv, partials=False):
        """Removes the time-shift from an observation's timing. This should be
        applied to observation event times, returning times applicable to
        external events.

        Input:
            time        a Scalar of observation event times.
            uv          a Pair defining the (u,v) coordinates at which the
                        time-shift applies.
            partials    if True, then the partial derivatives with respect to
                        the parameters are also returned as a subfield "d_dnav".
                        The partials are represented by a MatrixN with item
                        shape [1,N], where N is the number of parameters.

        Return:         a Vector3 of times applicable to external events.
        """

        # Default behavior is to do (almost) nothing
        time = Scalar.as_scalar(time)

        if partials:
            time.insert_subfield("d_dnav", self.dtime_dparams)

        return time

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Navigation(unittest.TestCase):

    def runTest(self):

        # TDB, but not much to test
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
