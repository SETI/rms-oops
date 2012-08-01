################################################################################
# oops_/cadence/reshaped.py: ReshapedCadence subclass of class Cadence
#
# 7/28/12 MRS - created and unit-tested.
################################################################################

import numpy as np
from oops_.array.all import *
from oops_.cadence.cadence_ import Cadence

class ReshapedCadence(Cadence):
    """ReshapedCadence is a Cadence subclass in which time steps are defined by
    another cadence with a different shape. This can be used, for example, to
    convert a 1-D cadence into an N-D cadence."""

    def __init__(self, cadence, shape):
        """Constructor for a ReshapedCadence.

        Input:
            cadence     the cadence to re-shape.
            shape       a tuple defining the new shape of the cadence.
        """

        self.cadence = cadence
        self.shape = tuple(shape)
        self.rank = len(self.shape)
        assert np.product(self.shape) == np.product(self.cadence.shape)

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime

        self.stride = np.cumproduct((self.shape + (1,))[::-1])[-2::-1]
                                                        # trust me, it works!

        self.oldshape = self.cadence.shape
        self.oldrank = len(self.cadence.shape)
        self.oldstride = np.cumproduct((self.oldshape + (1,))[::-1])[-2::-1]

        return

    @staticmethod
    def _reshape_tstep(tstep, oldshape, oldstride, oldrank,
                              newshape, newstride, newrank):
        """Private, static method to perform translations of tstep between the
        new and the old shapes of the cadence."""

        if oldrank == 1:
            tstep = Scalar.as_scalar(tstep)
        else:
            tstep = Tuple.as_tuple(tstep)

        is_floating = tstep.is_floating()

        # Convert to integers if necessary
        if is_floating:
            tstep_int = tstep.int()
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
            returned_tstep = Tuple(indices, tstep.mask)

        # Add the fractional part if necessary
        if is_floating:
            returned_tstep = returned_tstep.float()
            returned_tstep.vals[...,-1] += frac

        return returned_tstep

    def _old_tstep_from_new(self, tstep):
        """Private method to convert tsteps in the new stride to tsteps in the
        original stride."""

        return ReshapedCadence._reshape_tstep(tstep,
                                   self.shape, self.stride, self.rank,
                                   self.oldshape, self.oldstride, self.oldrank)

    def _new_tstep_from_old(self, tstep):
        """Private method to convert tsteps in the original stride to tsteps in
        the new stride."""

        return ReshapedCadence._reshape_tstep(tstep,
                                   self.oldshape, self.oldstride, self.oldrank,
                                   self.shape, self.stride, self.rank)

    def time_at_tstep(self, tstep, mask=False):
        """Returns the time associated with the given time step. This method
        supports non-integer step values.

        Input:
            tstep       a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         a Scalar of times in seconds TDB.
        """

        return self.cadence.time_at_tstep(self._old_tstep_from_new(tstep), mask)

    def time_range_at_tstep(self, tstep, mask=False):
        """Returns the range of time associated with the given integer time
        step index.

        Input:
            indices     a Scalar time step index or a Pair or Tuple of indices.
            mask        True to mask values outside the time limits.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        return self.cadence.time_range_at_tstep(self._old_tstep_from_new(tstep),
                                                mask)

    def tstep_at_time(self, time, mask=False):
        """Returns a the Scalar time step index or a Pair or Tuple of indices
        associated with a time in seconds TDB.

        Input:
            time        a Scalar of times in seconds TDB.
            mask        True to mask time values not sampled within the cadence.

        Return:         a Scalar, Pair or Tuple of time step indices.
        """

        return self._new_tstep_from_old(self.cadence.tstep_at_time(time, mask))

    def time_is_inside(self, time, inclusive=True):
        """Returns a boolean Numpy array indicating which elements in a given
        Scalar of times fall inside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to include the end moment of a time interval;
                        False to exclude.

        Return:         a boolean Numpy array indicating which time values are
                        sampled by the cadence.
        """

        return self.cadence.time_is_inside(time, inclusive)

    def time_shift(self, secs):
        """Returns a duplicate of the given cadence, with all times shifted by
        a specified number of seconds."

        Input:
            secs        the number of seconds to shift the time later.
        """

        return ReshapedCadence(self.cadence.time_shift(secs), self.shape)

    def as_continuous(self):
        """Returns a shallow copy of the given cadence, with equivalent strides
        but with the property that the cadence is continuous.
        """

        return ReshapedCadence(self.cadence.time_shift(secs), self.shape)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_ReshapedCadence(unittest.TestCase):

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

        if arg.is_integer():
            self.assertTrue(arg2.is_integer())
        else:
            self.assertTrue(arg2.is_floating())

    def runTest(self):

        self.TEST((10,), (2,5), Scalar(1))
        self.TEST((10,), (2,5), Scalar(1.5))
        self.TEST((10,), (2,5), Scalar(np.arange(10)))
        self.TEST((10,), (2,5), Scalar(np.arange(20)/2.))
        self.TEST((10,), (2,5), Scalar(np.arange(10).reshape(5,2)))
        self.TEST((10,), (2,5), Scalar((np.arange(20)/2.).reshape(2,5,2)))

        self.TEST((2,3,4), (24,), Tuple((1,2,3)))
        self.TEST((2,3,4), (24,), Tuple((1,2,3.5)))
        self.TEST((2,3,4), (24,), Tuple([(1,2,3),(1,2,3.5),(0,0,0.25)]))

        self.TEST((2,3,4), (4,6), Tuple((1,2,3)))
        self.TEST((2,3,4), (4,6), Tuple((1,2,3.5)))
        self.TEST((2,3,4), (4,6), Tuple([(1,2,3),(1,2,3.5),(0,0,0.25)]))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################

