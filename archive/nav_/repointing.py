################################################################################
# oops/nav_/repointing.py: Subclass Repointing of class Navigation
################################################################################

import numpy as np

from polymath             import *
from oops.nav_.navigation import Navigation

#*******************************************************************************
# Repointing
#*******************************************************************************
class Repointing(Navigation):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A Repointing is a Navigation subclass that describes a pointing
    correction to an observation via a set of two or three rotation angles.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, angles):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """Constructor for a Navigation object.

        Input:
            angles      a tuple or list of two or three angles in radians.
                dx      an offset along the x-axis, defined as a small rotation
                        about the y-axis.
                dy      an offset along the y-axis, defined as a small rotation
                        about the x-axis.
                theta   an optional rotation about the z-axis.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        angles = np.array(angles)
        assert len(angles.shape) == 1

        self.nparams = angles.size
        assert self.nparams in (2,3)

        if self.nparams == 2:
            self.axes = (0,1)
        else:
            self.axes = (2,0,1)

        self.set_params(angles)
    #===========================================================================



    #===========================================================================
    # set_params
    #===========================================================================
    def set_params(self, angles):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Part of the Fittable interface. Re-defines the navigation given a
        new set of parameters.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------------------

        # Internal method for a rotation matrix about one axis
        #---------------------------------------------------------
        def rotation_matrix(axis, angle):
            axis2 = axis
            axis0 = (axis2 + 1) % 3
            axis1 = (axis2 + 2) % 3

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            matrix = np.zeros((3,3))
            matrix[axis2, axis2] = 1.
            matrix[axis0, axis0] =  cos_angle
            matrix[axis1, axis1] =  cos_angle
            matrix[axis0, axis1] =  sin_angle
            matrix[axis1, axis0] = -sin_angle

            dmatrix_dparam = np.zeros((3,3))
            dmatrix_dparam[axis0, axis0] = -sin_angle
            dmatrix_dparam[axis1, axis1] = -sin_angle
            dmatrix_dparam[axis0, axis1] =  cos_angle
            dmatrix_dparam[axis1, axis0] = -cos_angle

            return (Matrix3(matrix), MatrixN(dmatrix_dparam))

        self.angles = np.array(angles)
        assert self.angles.shape == (self.nparams,)

        (self.matrix,
         dmatrix_dparam) = rotation_matrix(self.axes[0], angles[0])
        dmatrix_dparams = [dmatrix_dparam]

        for i in range(1,self.nparams):
            (matrix,
             dmatrix_dparam) = rotation_matrix(self.axes[i], angles[i])
            for j in range(len(dmatrix_dparams)):
                dmatrix_dparams[j] *= matrix

            dmatrix_dparams += [self.matrix * dmatrix_dparam]
            self.matrix *= matrix

        for i in range(self.nparams):
            dmatrix_dparams[i] = dmatrix_dparams[i].vals

        self.dmatrix_dparams = MatrixN(dmatrix_dparams)
        # dmatrix_dparams.shape = [N], item = [3,3]

        #--------------------------------
        # For the inverse rotation...
        #--------------------------------
        self.matrixT = self.matrix.T()
        self.dmatrixT_dparams = self.dmatrix_dparams.T()
    #===========================================================================



    #===========================================================================
    # get_params
    #===========================================================================
    def get_params(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Part of the Fittable interface. Returns the current parameters.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.angles
    #===========================================================================



    #===========================================================================
    # copy
    #===========================================================================
    def copy(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Part of the Fittable interface. Returns a deep copy of the object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        return Repointing(self.get_params().copy())
    #===========================================================================



    #===========================================================================
    # distort
    #===========================================================================
    def distort(self, los, t=None, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Applies the distortion to line-of-sight vectors. This should be
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------------------------------
        # Rotate the line of sight, with optional time-derivatives
        #-------------------------------------------------------------
        distorted_los = self.matrix * los

        #------------------------------------------------------------------
        # Fill in the partial derivatives if needed
        #------------------------------------------------------------------
        if partials:
            los_reshaped = los.append_axes(1)   # so [...,1] * [nparams] works
            dlos_swapped = self.dmatrix_dparams * los_reshaped
            dlos_dparams = MatrixN(dlos_swapped.vals.swapaxes(-2,-1))
            distorted_los.insert_subfield("d_dnav", dlos_dparams)

        return distorted_los
    #===========================================================================



    #===========================================================================
    # undistort
    #===========================================================================
    def undistort(self, los, t=None, partials=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Removes the distortion from the line-of-sight vectors. This should
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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #---------------------------------------------------------------
        # Un-distort the line of sight, with optional time-derivatives
        #---------------------------------------------------------------
        undistorted_los = self.matrixT * los

        #----------------------------------------------
        # Fill in the partial derivatives if needed
        #----------------------------------------------
        if partials:
            los_reshaped = los.append_axes(1)   # so [...,1] * [nparams] works
            dlos_swapped = self.dmatrixT_dparams * los_reshaped
            dlos_dparams = MatrixN(dlos_swapped.vals.swapaxes(-2,-1))
            undistorted_los.insert_subfield("d_dnav", dlos_dparams)

        return undistorted_los
    #===========================================================================



################################################################################
# UNIT TESTS
################################################################################

import unittest
import numpy.random as random

#*******************************************************************************
# Test_Repointing
#*******************************************************************************
class Test_Repointing(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        los = Vector3(random.randn(200,10,3))

        #--------------------------
        # 2-axis navigation
        #--------------------------
        angles = np.array((-0.15,0.25))
        nav = Repointing(angles)
        distorted_los = nav.distort(los)
        test_los = nav.undistort(distorted_los)
        self.assertTrue(abs(test_los - los) < 1.e-14)

        undistorted_los = nav.undistort(los)
        test_los = nav.distort(undistorted_los)
        self.assertTrue(abs(test_los - los) < 1.e-14)

        #--------------------------
        # 3-axis navigation
        #--------------------------
        angles = np.array((0.1,0.2,0.3))
        nav = Repointing(angles)
        distorted_los = nav.distort(los)
        test_los = nav.undistort(distorted_los)
        self.assertTrue(abs(test_los - los) < 1.e-14)

        undistorted_los = nav.undistort(los)
        test_los = nav.distort(undistorted_los)
        self.assertTrue(abs(test_los - los) < 1.e-14)

        #--------------------------
        # 2-axis derivatives
        #--------------------------
        DELTA = 1.e-7
        angles = np.array((-0.15,0.35))
        nav = Repointing(angles)
        distorted_los = nav.distort(los, partials=True)
        undistorted_los = nav.undistort(los, partials=True)

        for i in range(angles.size):
            angles_lo = angles.copy()
            angles_hi = angles.copy()

            angles_lo[i] -= DELTA
            angles_hi[i] += DELTA

            nav_lo = Repointing(angles_lo)
            nav_hi = Repointing(angles_hi)

            distorted_lo = nav_lo.distort(los)
            distorted_hi = nav_hi.distort(los)

            d_los_d_param = (distorted_hi - distorted_lo) / (2*DELTA)
            diff = d_los_d_param - Vector3(distorted_los.d_dnav.vals[...,i])
            self.assertTrue(abs(diff) < 1.e-7)

            undistorted_lo = nav_lo.undistort(los)
            undistorted_hi = nav_hi.undistort(los)

            d_los_d_param = (undistorted_hi - undistorted_lo) / (2*DELTA)
            diff = d_los_d_param - Vector3(undistorted_los.d_dnav.vals[...,i])
            self.assertTrue(abs(diff) < 1.e-7)

        #--------------------------
        # 3-axis derivatives
        #--------------------------
        DELTA = 1.e-7
        angles = np.array((-0.25,0.4,-0.1))
        nav = Repointing(angles)
        distorted_los = nav.distort(los, partials=True)
        undistorted_los = nav.undistort(los, partials=True)

        for i in range(0,angles.size):
            angles_lo = angles.copy()
            angles_hi = angles.copy()

            angles_lo[i] -= DELTA
            angles_hi[i] += DELTA

            nav_lo = Repointing(angles_lo)
            nav_hi = Repointing(angles_hi)

            distorted_lo = nav_lo.distort(los)
            distorted_hi = nav_hi.distort(los)

            d_los_d_param = (distorted_hi - distorted_lo) / (2*DELTA)
            diff = d_los_d_param - Vector3(distorted_los.d_dnav.vals[...,i])
            self.assertTrue(abs(diff) < 1.e-7)

            undistorted_lo = nav_lo.undistort(los)
            undistorted_hi = nav_hi.undistort(los)

            d_los_d_param = (undistorted_hi - undistorted_lo) / (2*DELTA)
            diff = d_los_d_param - Vector3(undistorted_los.d_dnav.vals[...,i])
            self.assertTrue(abs(diff) < 1.e-7)
    #===========================================================================



#*******************************************************************************



#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
