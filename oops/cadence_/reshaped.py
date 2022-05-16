################################################################################
# oops/cadence_/reshaped.py: ReshapedCadence subclass of class Cadence
################################################################################

import numpy as np
from polymath import *
from oops.cadence_.cadence import Cadence

#*******************************************************************************
# ReshapedCadence
#*******************************************************************************
class ReshapedCadence(Cadence):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    ReshapedCadence is a Cadence that has been reshaped.
    
    The time steps are defined by another cadence with a different shape.
    This can be used, for example, to convert a 1-D cadence into an N-D cadence.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    PACKRAT_ARGS = ['cadence', 'shape']

    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, cadence, shape):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Constructor for a ReshapedCadence.

        Input:
            cadence     the cadence to re-shape.
            shape       a tuple defining the new shape of the cadence.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.cadence = cadence
        self.shape = tuple(shape)
        self.rank = len(self.shape)
        assert np.product(self.shape) == np.product(self.cadence.shape)

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime
        self.lasttime = self.cadence.lasttime
        self.is_continuous = cadence.is_continuous

        self.stride = np.cumproduct((self.shape + (1,))[::-1])[-2::-1]
                                                        # trust me, it works!

        self.oldshape = self.cadence.shape
        self.oldrank = len(self.cadence.shape)
        self.oldstride = np.cumproduct((self.oldshape + (1,))[::-1])[-2::-1]

        return
    #===========================================================================



    #===========================================================================
    # _reshape_tstep
    #===========================================================================
    @staticmethod
    def _reshape_tstep(tstep, oldshape, oldstride, oldrank,
                              newshape, newstride, newrank):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Perform translations between new and old shapes of the cadence.
	"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if oldrank == 1:
            tstep = Scalar.as_scalar(tstep)
        else:
            tstep = Vector.as_vector(tstep)

        is_floating = tstep.is_float()

        # Convert to integers if necessary
        if is_floating:
            tstep_int = tstep.as_int()
            if oldrank == 1:
                frac = tstep.vals - tstep_int.vals
            else:
                frac = tstep.vals[...,-1] - tstep_int.vals[...,-1]
        else:
            tstep_int = tstep
            frac = 0

        # Convert the integer tstep to an offset from the first element
        if oldrank == 1:
            offset = tstep_int.vals
        else:
            offset = np.sum(oldstride * tstep_int.vals, axis=-1)

        # If the conversion is to a cadence of rank one, we're (almost) done
        if newrank == 1:
            return Scalar(offset + frac, tstep.mask)

        # Convert the integer offset to an integer index in the new stride
        # Trust me, this works
        offset = np.array(offset)
        offset = offset.reshape(offset.shape + (1,))
        indices = (offset // newstride) % newshape

        # Convert indices to the proper class
        if newrank == 2:
            returned_tstep = Pair(indices, tstep.mask)
        else:
            returned_tstep = Vector(indices, tstep.mask)

        # Add the fractional part if necessary
        if is_floating:
            returned_tstep = returned_tstep.as_float()
            returned_tstep.vals[...,-1] += frac

        return returned_tstep
    #===========================================================================



    #===========================================================================
    # _old_tstep_from_new
    #===========================================================================
    def _old_tstep_from_new(self, tstep):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Convert tsteps in the new stride to the original stride.
	"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return ReshapedCadence._reshape_tstep(tstep,
                                   self.shape, self.stride, self.rank,
                                   self.oldshape, self.oldstride, self.oldrank)
    #===========================================================================



    #===========================================================================
    # _new_tstep_from_old
    #===========================================================================
    def _new_tstep_from_old(self, tstep):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Convert tsteps in the original stride the new stride.
	"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return ReshapedCadence._reshape_tstep(tstep,
                                   self.oldshape, self.oldstride, self.oldrank,
                                   self.shape, self.stride, self.rank)
    #===========================================================================



    #===========================================================================
    # time_at_tstep
    #===========================================================================
    def time_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return the time(s) associated with the given time step(s).
        
        This method supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.cadence.time_at_tstep(self._old_tstep_from_new(tstep), mask)
    #===========================================================================



    #===========================================================================
    # time_range_at_tstep
    #===========================================================================
    def time_range_at_tstep(self, tstep, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return the range of time(s) for the given integer time step(s).

        Input:
            indices     a Scalar time step index or a Pair of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.cadence.time_range_at_tstep(self._old_tstep_from_new(tstep),
                                                mask)
    #===========================================================================



    #===========================================================================
    # tstep_at_time
    #===========================================================================
    def tstep_at_time(self, time, mask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return the time step(s) for given time(s).

        This method supports non-integer time values.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar or Pair of time step indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self._new_tstep_from_old(self.cadence.tstep_at_time(time, mask))
    #===========================================================================



    #===========================================================================
    # time_is_inside
    #===========================================================================
    def time_is_inside(self, time, inclusive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
	Return which time(s) fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a Boolean array indicating which time values are
                        sampled by the cadence. A masked time results in a
                        value of False, not a masked Boolean.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.cadence.time_is_inside(time, inclusive)
    #===========================================================================



    #===========================================================================
    # time_shift
    #===========================================================================
    def time_shift(self, secs):
        """Return a duplicate with all times shifted by given amount."

        Input:
            secs        the number of seconds to shift the time later.
        """

        return ReshapedCadence(self.cadence.time_shift(secs), self.shape)
    #===========================================================================



    #===========================================================================
    # as_continuous
    #===========================================================================
    def as_continuous(self):
        """Return a shallow copy forced to be continuous.
        
        For Sequence this is accomplished by forcing the exposure times to
        be equal to the stride for each step.
        """

        return ReshapedCadence(self.cadence.as_continuous(), self.shape)
    #===========================================================================


#*******************************************************************************



################################################################################
# UNIT TESTS
################################################################################

import unittest

#*******************************************************************************
# Test_ReshapedCadence
#*******************************************************************************
class Test_ReshapedCadence(unittest.TestCase):

    #===========================================================================
    # TEST
    #===========================================================================
    # A complete test there-and-back of _reshape_tstep()
    def TEST(self, oldshape, newshape, arg):

        oldstride = np.cumproduct((oldshape + (1,))[::-1])[-2::-1]
        newstride = np.cumproduct((newshape + (1,))[::-1])[-2::-1]
        oldrank = len(oldshape)
        newrank = len(newshape)

        arg1 = ReshapedCadence._reshape_tstep(arg,
                                              oldshape, oldstride, oldrank,
                                              newshape, newstride, newrank)
        arg2 = ReshapedCadence._reshape_tstep(arg1,
                                              newshape, newstride, newrank,
                                              oldshape, oldstride, oldrank)

        self.assertEqual(arg, arg2)

        self.assertEqual(type(arg), type(arg2))

        if arg.is_int():
            self.assertTrue(arg2.is_int())
        else:
            self.assertTrue(arg2.is_float())
    #===========================================================================



    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.cadence_.metronome import Metronome
    
        self.TEST((10,), (10,), Scalar(1))
        self.TEST((10,), (2,5), Scalar(1))
        self.TEST((10,), (2,5), Scalar(1.5))
        self.TEST((10,), (2,5), Scalar(np.arange(10)))
        self.TEST((10,), (2,5), Scalar(np.arange(20)/2.))
        self.TEST((10,), (2,5), Scalar(np.arange(10).reshape(5,2)))
        self.TEST((10,), (2,5), Scalar((np.arange(20)/2.).reshape(2,5,2)))

        self.TEST((2,3,4), (24,), Vector((1,2,3)))
        self.TEST((2,3,4), (24,), Vector((1,2,3.5)))
        self.TEST((2,3,4), (24,), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

        self.TEST((2,3,4), (4,6), Vector((1,2,3)))
        self.TEST((2,3,4), (4,6), Vector((1,2,3.5)))
        self.TEST((2,3,4), (4,6), Vector([(1,2,3),(1,2,3.5),(0,0,0.25)]))

        cadence = Metronome(100., 10., 10., 100)
        reshaped = ReshapedCadence(cadence, (25,4))
        self.assertTrue(reshaped.is_continuous)
        self.assertEqual(reshaped.time_at_tstep((0,0)), 100.)
        self.assertEqual(reshaped.time_at_tstep((0,1)), 110.)
        self.assertEqual(reshaped.time_at_tstep((1,0)), 140.)
        self.assertEqual(reshaped.time_at_tstep((1,1)), 150.)
        self.assertEqual(reshaped.time_at_tstep((1,1.5)), 155.)

        cadence = Metronome(100., 15., 10., 100)
        reshaped = ReshapedCadence(cadence, (25,4))
        self.assertFalse(reshaped.is_continuous)
        self.assertEqual(reshaped.time_at_tstep((0,0)), 100.)
        self.assertEqual(reshaped.time_at_tstep((0,1)), 115.)
        self.assertEqual(reshaped.time_at_tstep((1,0)), 160.)
        self.assertEqual(reshaped.time_at_tstep((1,1)), 175.)
        self.assertEqual(reshaped.time_at_tstep((1,1.5)), 180.)

        new_cadence = reshaped.as_continuous()
        self.assertTrue(new_cadence.is_continuous)
        self.assertEqual(new_cadence.time_at_tstep((0,0)), 100.)
        self.assertEqual(new_cadence.time_at_tstep((0,1)), 115.)
        self.assertEqual(new_cadence.time_at_tstep((1,0)), 160.)
        self.assertEqual(new_cadence.time_at_tstep((1,1)), 175.)
        self.assertEqual(new_cadence.time_at_tstep((1,1.5)), 182.5)
    #===========================================================================


#*******************************************************************************


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

