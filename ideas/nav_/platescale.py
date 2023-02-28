################################################################################
# oops/nav_/platescale.py: Subclass PlateScale of class Navigation.
################################################################################

import numpy as np

from polymath             import *
from oops.nav_.navigation import Navigation

class PlateScale(Navigation):
    """A PlateScale is a Navigation subclass that expands or contracts the x-
    and y-components of all line-of-sight vectors by constant factors.
    """

    #===========================================================================
    def __init__(self, scale):
        """Constructor for a PlateScale object.

        Input:
            scale       one or two scale factors. If one, then it is applied to
                        both the x- and y-components of the vectors; if two,
                        then the x- and y-components are scaled independently.
        """

        scale = np.array(scale)
        if np.shape(scale) == ():       # allow a constant instead of a tuple
            scale = np.array((scale,))

        assert len(scale.shape) == 1

        self.nparams = scale.size
        assert self.nparams in (1,2)

        self.set_params(scale)

        self.xymat = np.array([[1.,0.,0.],
                               [0.,1.,0.],
                               [0.,0.,0.]])

    #===========================================================================
    def set_params(self, scale):
        """Part of the Fittable interface. Re-defines the navigation given a
        new set of parameters.
        """

        # Check the parameters
        scale = np.array(scale)
        if np.shape(scale) == ():       # allow a constant instead of a tuple
            scale = np.array((scale,))

        assert scale.shape == (self.nparams,)

        # Save the parameters
        self.scale = scale

        # Define a vector scale factor and its inverse
        if self.nparams == 1:
            self.vector = Vector3((scale[0], scale[0], 1.))
        else:
            self.vector = Vector3((scale[0], scale[1], 1.))

        self.vector.insert_subfield("d_dt", self.vector.plain())

        self.inverse = 1. / self.vector
        self.inverse.insert_subfield("d_dt", self.inverse.plain())

        # Prepare for the partial derivatives of the inverse distortion
        self.xyinv = np.array([[-self.inverse[0], 0., 0.],
                               [0., -self.inverse[1], 0.],
                               [0., 0., 0.]])

    #===========================================================================
    def get_params(self):
        """Part of the Fittable interface. Returns the current parameters.
        """

        return self.scale

    #===========================================================================
    def copy(self):
        """Part of the Fittable interface. Returns a deep copy of the object.
        """

        return PlateScale(self.get_params().copy())

    #===========================================================================
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
                        subfield "d_nav" with item shape [3,N], where N is the
                        number of parameters.

        Return          a Vector3 of un-distorted vectors. If the input Vector3
                        has a subfield "d_dt", this is also distorted and
                        returned as a subfield.
        """

        # Distort the line of sight, with optional time-derivatives
        distorted_los = los * self.vector

        # Fill in the partial derivatives if needed
        if partials:
            dlos_dparams = MatrixN(los.vals[...,np.newaxis] * self.xymat)
            distorted_los.insert_subfield("d_dnav", dlos_dparams)

        return distorted_los

    #===========================================================================
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
                        subfield "d_dnav" with item shape [3,N], where N is the
                        number of parameters.

        Return          a Vector3 of un-distorted vectors. If the input Vector3
                        has a subfield "d_dt", this is also un-distorted and
                        returned as a subfield.
        """

        # Un-distort the line of sight, with optional time-derivatives
        undistorted_los = los * self.inverse

        # Fill in the partial derivatives if needed
        if partials:
            dlos_dparams = MatrixN(los.vals[...,np.newaxis] * self.xyinv)
            undistorted_los.insert_subfield("d_dnav", dlos_dparams)

        return undistorted_los

################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

class Test_PlateScale(unittest.TestCase):

    def runTest(self):

        # TBD, testing needed!
        pass

#########################################
if __name__ == '__main__': # pragma: no cover
    unittest.main(verbosity=2)
################################################################################
