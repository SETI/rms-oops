################################################################################
# oops/fov_/subarray.py: Subarray subclass of FOV
################################################################################

import numpy as np
from polymath import *

from oops.fov_.fov import FOV

#*******************************************************************************
# Subarray class
#*******************************************************************************
class Subarray(FOV):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Subarray is a subclass of FOV that describes a rectangular region of a
    larger FOV.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    PACKRAT_ARGS = ['fov', 'new_los', 'uv_shape', 'uv_los']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, fov, new_los, uv_shape, uv_los=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Constructor for a Subarray.

        In the returned FOV object, the ICS origin and/or the optic axis have
        been modified.

        Input:
            fov         the FOV object within which this subarray is defined.

            new_los     a tuple or Pair defining the location of the subarray's
                        line of sight in the (u,v) coordinates of the original
                        FOV.

            uv_shape    a single value, tuple or Pair defining the new size of
                        the field of view in pixels.

            uv_los      a single value, tuple or Pair defining the (u,v)
                        coordinates of the new line of sight. By default,
                        this is the midpoint of the rectangle, i.e, uv_shape/2.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
        self.new_origin_in_old_uv.as_readonly

        # Required fields
        self.uv_scale = self.fov.uv_scale
        self.uv_area  = self.fov.uv_area
    #===========================================================================



    #===========================================================================
    # xy_from_uv
    #===========================================================================
    def xy_from_uv(self, uv_pair, derivs=False, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return (u,v) FOV coordinates given (x,y) camera frame coordinates.

        If derivs is True, then any derivatives in (x,y) get propagated into
        the (u,v) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        old_xy = self.fov.xy_from_uv(self.new_origin_in_old_uv + uv_pair,
                                     derivs, **keywords)
        return old_xy - self.new_los_wrt_old_xy
    #===========================================================================



    #===========================================================================
    # uv_from_xy
    #===========================================================================
    def uv_from_xy(self, xy_pair, derivs=False, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Return (x,y) camera frame coordinates given FOV coordinates (u,v).

        If derivs is True, then any derivatives in (u,v) get propagated into
        the (x,y) returned.

        Additional parameters that might affect the transform can be included
        as keyword arguments.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        old_uv = self.fov.uv_from_xy(self.new_los_wrt_old_xy + xy_pair,
                                     derivs, **keywords)
        return old_uv - self.new_origin_in_old_uv
    #===========================================================================


#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Subarray(unittest.TestCase):

    def runTest(self):

        # Imports just required for unit testing
        from oops.fov_.flatfov import FlatFOV

        flat = FlatFOV((1/2048.,-1/2048.), 101, (50,75))

        test = Subarray(flat, (50,75), 101, (50,75))
        buffer = np.empty((101,101,2))
        buffer[:,:,0] = np.arange(101).reshape(101,1)
        buffer[:,:,1] = np.arange(101)
        uv = Pair(buffer)

        xy = test.xy_from_uv(buffer)
        self.assertEqual(xy, flat.xy_from_uv(uv))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, uv)

        self.assertEqual(test.area_factor(uv), 1.)

        ############################

        test = Subarray(flat, (50,75), 51)
        buffer = np.empty((51,51,2))
        buffer[:,:,0] = np.arange(51).reshape(51,1) + 0.5
        buffer[:,:,1] = np.arange(51) + 0.5
        uv = Pair(buffer)

        xy = test.xy_from_uv(buffer)
        self.assertEqual(xy, -test.xy_from_uv(buffer[-1::-1,-1::-1]))

        uv_test = test.uv_from_xy(xy)
        self.assertEqual(uv_test, uv)

        self.assertEqual(test.area_factor(uv), 1.)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
