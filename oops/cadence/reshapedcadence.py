################################################################################
# oops/cadence/reshapedcadence.py: ReshapedCadence subclass of class Cadence
################################################################################

import numpy as np
from polymath     import Qube, Scalar, Pair, Vector
from oops.cadence import Cadence

class ReshapedCadence(Cadence):
    """A Cadence that has been reshaped.

    The time steps are defined by another cadence with a different shape.
    This can be used, for example, to convert a 1-D cadence into an N-D cadence.
    """

    #===========================================================================
    def __init__(self, cadence, shape):
        """Constructor for a ReshapedCadence.

        Input:
            cadence     the cadence to re-shape.
            shape       a tuple defining the new shape of the cadence.
        """

        self.cadence = cadence
        self.shape = tuple(shape)
        self._rank = len(self.shape)
        self._size = int(np.prod(self.shape))

        if self._size != np.prod(self.cadence.shape):
            raise ValueError('ReshapedCadence size and shape are incompatible')

        if self._rank > 2:
            raise ValueError('%d-D cadences are not supported' % self._rank)

        self.time = self.cadence.time
        self.midtime = self.cadence.midtime
        self.lasttime = self.cadence.lasttime
        self.is_continuous = self.cadence.is_continuous
        self.is_unique = self.cadence.is_unique
        self.min_tstride = self.cadence.min_tstride
        self.max_tstride = self.cadence.max_tstride

        self._stride = np.cumprod((self.shape + (1,))[::-1])[-2::-1]
                                                        # trust me, it works!

        self._old_shape = self.cadence.shape
        self._old_rank = len(self.cadence.shape)
        self._old_stride = np.cumprod((self._old_shape + (1,))[::-1])[-2::-1]

    def __getstate__(self):
        return (self.cadence, self.shape)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @staticmethod
    def _reshape_tstep(tstep, old_shape, old_stride, old_rank,
                              new_shape, new_stride, new_rank, size,
                              remask=False, derivs=False, inclusive=True):
        """Translate a cadence index from old shape to new."""

        # Convert old tstep to integer offset + fraction; remask for now
        if old_rank == 1:
            tstep = Scalar.as_scalar(tstep, recursive=derivs)
            index_1d = tstep.int(old_shape[0], remask=True,
                                 inclusive=inclusive, clip=True)
            remainder = tstep - index_1d
            frac = remainder.clip(0, 1, remask=True)
            index_1d = index_1d.vals
        else:
            tstep = Vector.as_vector(tstep, recursive=derivs)
            tstep_int = tstep.int(old_shape, remask=True, inclusive=inclusive,
                                             clip=True)
            remainder = (tstep - tstep_int).to_scalar(-1)
            frac = remainder.clip(0, 1, remask=True)
            index_1d = np.sum(old_stride * tstep_int.vals, axis=-1)

        # If the conversion is to a cadence of rank one, we're done
        if new_rank == 1:
            result = index_1d + frac
            if not remask:
                result = result.remask(tstep.mask)

            return result

        # Convert the offset to an integer index using the new stride
        # Trust me, this works
        new_offset = np.reshape(index_1d, np.shape(index_1d) + (1,))
        indices = (new_offset // new_stride) % new_shape

        # Convert to float if necessary
        if tstep.is_float():
            indices = np.asfarray(indices)

        # Restore fractional part
        indices[...,-1] += frac.vals

        # Select the new mask
        if remask:
            mask = frac.mask
        else:
            mask = tstep.mask

        # Convert indices to the proper class
        if new_rank == 2:
            class_ = Pair
        else:                   # not currently supported
            class_ = Vector

        new_tstep = class_(indices, mask)

        # Restore derivatives if necessary
        if derivs:
            for key, deriv in tstep.derivs.items():
                # Construct the new derivative

                # When a derivative has a numerator, the derivatives are always
                # required/expected to be zero along every axis except the last.

                shape = new_tstep.shape + new_tstep.numer + deriv.denom
                new_deriv_vals = np.zeros(shape)

                new_index = ((Ellipsis,) + (new_tstep.nrank-1) * (slice(None),)
                             + ((-1,) if new_tstep.nrank else ())
                             + deriv.drank * (slice(None),))

                old_index = ((Ellipsis,) + (deriv.nrank-1) * (slice(None),)
                             + ((-1,) if deriv.nrank else ())
                             + deriv.drank * (slice(None),))

                new_deriv_vals[new_index] = deriv.vals[old_index]
                    # Note that the above works if new_deriv_vals has no shape,
                    # because indx is an Ellipsis, which is a valid index, and
                    # because the constructor converts a shapeless array value
                    # to a scalar.

                # Prepare the new mask
                if isinstance(deriv.mask, (bool, np.bool_)):
                    new_deriv_mask = mask
                elif deriv.mask is tstep.mask:  # it's common for derivs
                                                # to share the parent's mask
                    new_deriv_mask = mask
                else:
                    new_deriv_mask = deriv.mask.reshape(new_tstep.shape)
                    new_deriv_mask = Qube.or_(mask, new_deriv_mask)

                # Construct and insert the new derivative
                new_deriv = class_(new_deriv_vals, new_deriv_mask,
                                   nrank=new_tstep.nrank, drank=deriv.drank)
                new_tstep.insert_deriv(key, new_deriv)

        return new_tstep

    #===========================================================================
    def _old_tstep_from_new(self, tstep, remask=False, derivs=False,
                                         inclusive=True):
        """Convert a tstep index for the old cadence to the new."""

        return ReshapedCadence._reshape_tstep(
                            tstep,
                            self.shape, self._stride, self._rank,
                            self._old_shape, self._old_stride, self._old_rank,
                            self._size,
                            remask=remask, derivs=derivs, inclusive=inclusive)

    #===========================================================================
    def _new_tstep_from_old(self, tstep, remask=False, derivs=False,
                                         inclusive=True):
        """Convert a tstep index for the new cadence to the old."""

        return ReshapedCadence._reshape_tstep(
                            tstep,
                            self._old_shape, self._old_stride, self._old_rank,
                            self.shape, self._stride, self._rank, self._size,
                            remask=remask, derivs=derivs, inclusive=inclusive)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        In multidimensional cadences, indexing beyond the dimensions of the
        cadence returns the time at the nearest edge of the cadence's shape.

        Input:
            tstep       a Scalar or Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = self._old_tstep_from_new(tstep, remask=remask, derivs=derivs,
                                                inclusive=inclusive)

        return self.cadence.time_at_tstep(tstep, remask=remask, derivs=derivs,
                                                 inclusive=inclusive)

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        tstep = self._old_tstep_from_new(tstep, derivs=False, remask=remask,
                                                inclusive=inclusive)

        return self.cadence.time_range_at_tstep(tstep, remask=remask,
                                                       inclusive=inclusive)

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of time in the returned
                        tstep.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar or Pair of time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # Converting to 1-D or continuous cadences, this is fairly easy...
        if self._rank == 1 or self.is_continuous:
            tstep = self.cadence.tstep_at_time(time, remask=remask,
                                               derivs=derivs,
                                               inclusive=inclusive)
            tstep = self._new_tstep_from_old(tstep, remask=remask,
                                             derivs=derivs,
                                             inclusive=inclusive)

        # Otherwise...
        else:

            # Remove the time mask and remask, so tstep is masked if and only if
            # a time is out of range, including in a gap between discontinuous
            # time steps.
            tstep = self.cadence.tstep_at_time(time.without_mask(), remask=True,
                                               derivs=derivs,
                                               inclusive=inclusive)
            tstep = self._new_tstep_from_old(tstep, remask=True, derivs=derivs,
                                             inclusive=inclusive)

            # For masked tsteps that have wrapped forward to the next line,
            # shift them back to the end of the previous line.

            # Note--this only works correctly for 2-D
            wrapped = (tstep.mask & (tstep.vals[...,-1] == 0)
                                  & (tstep.vals[...,-2] >  0))
            if np.shape(wrapped):
                tstep.vals[wrapped,-2] -= 1
                tstep.vals[wrapped,-1] = self.shape[-1]
            elif wrapped:
                tstep.vals[-2] -= 1
                tstep.vals[-1] = self.shape[-1]

            # Now repair the mask
            if remask:
                tstep = tstep.remask_or(time.mask)
            else:
                tstep = tstep.remask(time.mask)

        # Times beyond the end of a 2-D cadence require special handling
        if self._rank > 1:
            above = Qube.is_above(time, self.time[1], inclusive=inclusive)
            tstep[above] = type(tstep)(self.shape, remask)

        return tstep

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Scalar or Pair time step index for time range.
            tstep_max   maximum Scalar or Pair time step index for time range.

        All returned indices will be in the allowed range for the cadence,
        inclusive, regardless of mask. If the time is not inside the cadence,
        tstep_max < tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Mask here; update the mask later if necessary
        (old_tstep_min,
         old_tstep_max) = self.cadence.tstep_range_at_time(time, remask=True,
                                                           inclusive=inclusive)

        if self.shape == self._old_shape:
            return (old_tstep_min, old_tstep_max)

        # Calculate the number of tsteps in each old range
        if self._old_rank == 1:
            count = old_tstep_max.vals - old_tstep_min.vals
        else:
            diffs = old_tstep_max.vals - old_tstep_min.vals - 1
            count = np.sum(self._old_stride * diffs, axis=-1) + 1

        # Get the new minimum tstep
        new_tstep_min = self._new_tstep_from_old(old_tstep_min, remask=False,
                                                 derivs=False, inclusive=True)

        # Calculate the tstep offset for each new range
        if self._rank == 1:
            new_tstep_max = new_tstep_min + count

        else:
            index_1d = (np.sum(self._stride * new_tstep_min.vals, axis=-1)
                        + np.maximum(count-1, 0))
            max_vals = (index_1d[...,np.newaxis] // self._stride) % self.shape
            max_vals += 1

            # Handle count == 0, where last axis of range must have size 0
            mask = (count == 0)
            if np.shape(mask):
                max_vals[mask,-1] = new_tstep_min.vals[mask,-1]
            elif mask:
                max_vals[-1] = new_tstep_min.vals[-1]

            new_tstep_max = new_tstep_min.clone()
            new_tstep_max._set_values_(max_vals)

            # Make sure that the new tstep range will be continuous
            if not self.is_unique and self._rank > 1:
                multiple_rows = np.any((new_tstep_min.vals[...,:-1] !=
                                        new_tstep_max.vals[...,:-1] - 1))
                incomplete_lines = ((new_tstep_min.vals[...,-1] != 0) |
                                 (new_tstep_max.vals[...,-1] != self.shape[-1]))
                unmasked = new_tstep_min.antimask
                problems = multiple_rows & incomplete_lines & unmasked
                if np.any(problems):
                    if np.isscalar(problems):
                        timeval = time.vals
                        minval = new_tstep_min
                        maxval = new_tstep_max
                    else:
                        timeval = time[problems][0]
                        minval = new_tstep_min[problems][0]
                        maxval = new_tstep_max[problems][0]

                    raise ValueError('returned tstep range is discontinuous ' +
                                     'at %s: %s, %s' % (timeval, minval,
                                                                 maxval))

        # Make sure that the old tstep range was continuous
        if not self.is_unique and self._old_rank > 1:
            multiple_rows = np.any(old_tstep_min.vals[...,:-1] !=
                                   old_tstep_max.vals[...,:-1] - 1)
            incomplete_lines = ((old_tstep_min.vals[...,-1] != 0) |
                                (old_tstep_max.vals[...,-1] != self.shape[-1]))
            unmasked = old_tstep_min.antimask
            problems = multiple_rows & incomplete_lines & unmasked
            if np.any(problems):
                if np.isscalar(problems):
                    timeval = time.vals
                    minval = new_tstep_min
                    maxval = new_tstep_max
                else:
                    timeval = time[problems][0]
                    minval = old_tstep_min[problems][0]
                    maxval = old_tstep_max[problems][0]

                raise ValueError('input tstep range is discontinuous at ' +
                                 '%s: %s, %s' % (timeval, minval, maxval))

        # Restore the original mask if necessary
        if not remask:
            new_tstep_min = new_tstep_min.remask(time.mask)
            new_tstep_max = new_tstep_max.remask(time.mask)

        return (new_tstep_min, new_tstep_max)

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Boolean mask indicating which time values are not
                        sampled by the cadence.
        """

        return self.cadence.time_is_outside(time, inclusive=inclusive)

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return ReshapedCadence(self.cadence.time_shift(secs), self.shape)

    #===========================================================================
    def as_continuous(self):
        """A shallow copy of this cadence, forced to be continuous.

        For Sequence this is accomplished by forcing the exposure times to
        be equal to the stride for each step.
        """

        return ReshapedCadence(self.cadence.as_continuous(), self.shape)

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.cadence.dualcadence import case_dual_metronome

class Test_ReshapedCadence(unittest.TestCase):

    # A complete test there-and-back of _reshape_tstep()

    def TEST(self, oldshape, newshape, arg):

        oldstride = np.cumprod((oldshape + (1,))[::-1])[-2::-1]
        newstride = np.cumprod((newshape + (1,))[::-1])[-2::-1]
        oldrank = len(oldshape)
        newrank = len(newshape)

        arg1 = ReshapedCadence._reshape_tstep(arg,
                                              oldshape, oldstride, oldrank,
                                              newshape, newstride, newrank,
                                              np.prod(oldshape))
        arg2 = ReshapedCadence._reshape_tstep(arg1,
                                              newshape, newstride, newrank,
                                              oldshape, oldstride, oldrank,
                                              np.prod(oldshape))

        self.assertEqual(arg, arg2)

        self.assertEqual(type(arg), type(arg2))

        if arg.is_int():
            self.assertTrue(arg2.is_int())
        else:
            self.assertTrue(arg2.is_float())

    def runTest(self):

        from oops.cadence.metronome import Metronome

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

        ########################################################################
        # Compare a Metronome reshaped to 2-D to an equivalent DualCadence
        # cad1d: 100-101, 102-103, 104-105, ... 198-199.

        cad1d = Metronome(100., 2., 1., 50)

#         long = Metronome(100., 10., 1., 10)
#         short = Metronome(0, 2., 1., 5)
#         cad2d = DualCadence(long, short)
        cad2d = ReshapedCadence(cad1d, (10,5))

        case_dual_metronome(self, cad1d, cad2d)

        ############################################
        # Weirdly reshaped case, 100 -> (25,4)
        # 100-110, 110-120, 120-130, ...
        ############################################

        cadence = Metronome(100., 10., 10., 100)
        reshaped = ReshapedCadence(cadence, (25,4))

        self.assertTrue(reshaped.is_continuous)
        self.assertTrue(reshaped.is_unique)

        self.assertEqual(reshaped.time_at_tstep((0,0)), 100.)
        self.assertEqual(reshaped.time_at_tstep((0,1)), 110.)
        self.assertEqual(reshaped.time_at_tstep((1,0)), 140.)
        self.assertEqual(reshaped.time_at_tstep((1,1)), 150.)
        self.assertEqual(reshaped.time_at_tstep((1,1.5)), 155.)

        tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]])
        self.assertEqual(reshaped.time_at_tstep(tstep), [[100,110],[140,150]])

        tstep = Pair([[(0,0),(0,1)],[(1,0),(1,1)]], [[1,0],[0,0]])
        time = reshaped.time_at_tstep(tstep)
        self.assertTrue(np.all(tstep.mask == time.mask))
        self.assertEqual(time[0,0], Scalar.MASKED)
        self.assertEqual(reshaped.time_at_tstep(tstep), [[Scalar.MASKED,110],[140,150]])

        self.assertEqual(reshaped.tstep_at_time(100.), (0,0))
        self.assertEqual(reshaped.tstep_at_time(110.), (0,1))
        self.assertEqual(reshaped.tstep_at_time(140.), (1,0))
        self.assertEqual(reshaped.tstep_at_time(150.), (1,1))
        self.assertEqual(reshaped.tstep_at_time(155.), (1,1.5))

        for i in np.arange(-2., 28., 0.5):
          for j in np.arange(-1., 5., 0.25):
            k = 4*np.floor(i) + j
            (time1a, time1b) = cadence.time_range_at_tstep(k)
            (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                            inclusive=False)
            if not time2a.mask:
                self.assertEqual(time1a, time2a)
                self.assertEqual(time1b, time2b)

                time = reshaped.time_at_tstep((i,j), remask=True)
                tstep = reshaped.tstep_at_time(time)
                self.assertEqual(tstep, (np.floor(i),j))

        ############################################
        # Weirdly reshaped case, 100 -> (25,4), discontinuous
        # [100-108, 116-124, 132-140, 148-156], [164-172, ...], [..., 1684-1692]
        ############################################

        cadence = Metronome(100., 16., 8., 100)
        reshaped = ReshapedCadence(cadence, (25,4))
        self.assertFalse(reshaped.is_continuous)
        self.assertTrue(reshaped.is_unique)
        self.assertEqual(reshaped.time_at_tstep((0,0)), 100.)
        self.assertEqual(reshaped.time_at_tstep((0,1)), 116.)
        self.assertEqual(reshaped.time_at_tstep((1,0)), 164.)
        self.assertEqual(reshaped.time_at_tstep((1,1)), 180.)
        self.assertEqual(reshaped.time_at_tstep((1,1.5)), 184.)
        self.assertEqual(reshaped.time_at_tstep((1.5,1.5)), 184.)

        new_cadence = reshaped.as_continuous()
        self.assertTrue(new_cadence.is_continuous)
        self.assertEqual(new_cadence.time_at_tstep((0,0)), 100.)
        self.assertEqual(new_cadence.time_at_tstep((1,0)), 164.)
        self.assertEqual(new_cadence.time_at_tstep((1,1)), 180.)
        self.assertEqual(new_cadence.time_at_tstep((1,1.5)), 188)
        self.assertEqual(new_cadence.time_at_tstep((1.5,1.5)), 188)

        self.assertEqual(reshaped.tstep_at_time( 99.), (0,0))
        self.assertEqual(reshaped.tstep_at_time( 99., remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(100.), (0,0))
        self.assertEqual(reshaped.tstep_at_time(106.), (0,0.75))
        self.assertEqual(reshaped.tstep_at_time(116.), (0,1))
        self.assertEqual(reshaped.tstep_at_time(108.), (0,1))
        self.assertEqual(reshaped.tstep_at_time(108., remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(132.), (0,2))
        self.assertEqual(reshaped.tstep_at_time(148.), (0,3))
        self.assertEqual(reshaped.tstep_at_time(140.), (0,3))
        self.assertEqual(reshaped.tstep_at_time(140., remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(155.), (0,3.875))
        self.assertEqual(reshaped.tstep_at_time(156.), (0,4))
        self.assertEqual(reshaped.tstep_at_time(156., remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(163.99999), (0,4))
        self.assertEqual(reshaped.tstep_at_time(163.99999, remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(164.), (1,0))
        self.assertEqual(reshaped.tstep_at_time(1684), (24,3))
        self.assertEqual(reshaped.tstep_at_time(1692), (24,4))
        self.assertEqual(reshaped.tstep_at_time(1692, inclusive=False), (25,4))
        self.assertEqual(reshaped.tstep_at_time(1692, inclusive=False,
                                                      remask=True), Pair.MASKED)

        self.assertEqual(reshaped.tstep_range_at_time( 99.), ((0,0), (1,0)))
        self.assertEqual(reshaped.tstep_range_at_time( 99., remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(100.), ((0,0), (1,1)))
        self.assertEqual(reshaped.tstep_range_at_time(106.), ((0,0), (1,1)))
        self.assertEqual(reshaped.tstep_range_at_time(108.), ((0,0), (1,0)))
        self.assertEqual(reshaped.tstep_range_at_time(108., remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(115.999), ((0,0), (1,0)))
        self.assertEqual(reshaped.tstep_range_at_time(115.999, remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(116.), ((0,1), (1,2)))
        self.assertEqual(reshaped.tstep_range_at_time(148.), ((0,3), (1,4)))
        self.assertEqual(reshaped.tstep_range_at_time(140.), ((0,2), (1,2)))
        self.assertEqual(reshaped.tstep_range_at_time(140., remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(156.), ((0,3), (1,3)))
        self.assertEqual(reshaped.tstep_range_at_time(156., remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(163.999), ((0,3), (1,3)))
        self.assertEqual(reshaped.tstep_range_at_time(163.999, remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(164.), ((1,0), (2,1)))
        self.assertEqual(reshaped.tstep_range_at_time(1684), ((24,3), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1692), ((24,3), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1692, inclusive=False), ((24,3), (25,3)))
        self.assertEqual(reshaped.tstep_range_at_time(1692, inclusive=False,
                                                      remask=True)[0], Pair.MASKED)

        for i in np.arange(-2., 28., 0.5):
          for j in np.arange(-1., 5., 0.25):
            k = 4*np.floor(i) + j
            (time1a, time1b) = cadence.time_range_at_tstep(k)
            (time2a, time2b) = reshaped.time_range_at_tstep((i,j), remask=True,
                                                            inclusive=False)
            if not time2a.mask:
                self.assertEqual(time1a, time2a)
                self.assertEqual(time1b, time2b)

                time = reshaped.time_at_tstep((i,j), remask=True)
                tstep = reshaped.tstep_at_time(time)
                self.assertEqual(tstep, (np.floor(i),j))

        ############################################
        # Weirdly reshaped case, 100 -> (25,4), overlapping
        # [100-116, 110-126, 120-136, 130-146], [140-156, ...], [..., 1090-1106]
        ############################################

        cadence = Metronome(100., 10., 16., 100)
        reshaped = ReshapedCadence(cadence, (25,4))
        self.assertTrue(reshaped.is_continuous)
        self.assertFalse(reshaped.is_unique)
        self.assertEqual(reshaped.time_at_tstep((0,0)), 100.)
        self.assertEqual(reshaped.time_at_tstep((0,1)), 110.)
        self.assertEqual(reshaped.time_at_tstep((1,0)), 140.)
        self.assertEqual(reshaped.time_at_tstep((1,1)), 150.)
        self.assertEqual(reshaped.time_at_tstep((1,1.5)), 158.)
        self.assertEqual(reshaped.time_at_tstep((1.5,1.5)), 158.)

        tstep = Pair([(0,0),(0,1),(1,0),(1,1),(1,1.5),(1.5,1.5)],
                     [True] + 5*[False])
        self.assertEqual(reshaped.time_at_tstep(tstep),
                         (Scalar.MASKED, 110, 140, 150, 158, 158))
        tstep.insert_deriv('t' , Pair(np.arange(12).reshape(6,2)))
        tstep.insert_deriv('xy', Pair(np.ones((6,2,2)), drank=1))
        self.assertEqual(reshaped.time_at_tstep(tstep),
                         (Scalar.MASKED, 110, 140, 150, 158, 158))
        self.assertEqual(reshaped.time_at_tstep(tstep, derivs=True).d_dt,
                         (Scalar.MASKED, 48, 80, 112, 144, 176))
        self.assertEqual(reshaped.time_at_tstep(tstep, derivs=True).d_dxy,
                         Scalar(16 * np.ones((6,2)), [True] + 5*[False], drank=1))

        self.assertEqual(reshaped.time_range_at_tstep((0,0)), (100, 116))
        self.assertEqual(reshaped.time_range_at_tstep((0,1)), (110, 126))
        self.assertEqual(reshaped.time_range_at_tstep((1,0)), (140, 156))
        self.assertEqual(reshaped.time_range_at_tstep((1,1)), (150, 166))
        self.assertEqual(reshaped.time_range_at_tstep((1,1.5)), (150, 166))
        self.assertEqual(reshaped.time_range_at_tstep((1.5,1.5)), (150, 166))

        self.assertEqual(reshaped.time_range_at_tstep(tstep),
                         ((Scalar.MASKED, 110, 140, 150, 150, 150),
                          (Scalar.MASKED, 126, 156, 166, 166, 166)))

        self.assertEqual(reshaped.tstep_at_time( 99.), (0,0))
        self.assertEqual(reshaped.tstep_at_time( 99., remask=True), Pair.MASKED)
        self.assertEqual(reshaped.tstep_at_time(100.), (0,0))
        self.assertEqual(reshaped.tstep_at_time(108.), (0,0.5))
        self.assertEqual(reshaped.tstep_at_time(110.), (0,1))
        self.assertEqual(reshaped.tstep_at_time(140.), (1,0))
        self.assertEqual(reshaped.tstep_at_time(1090), (24,3))
        self.assertEqual(reshaped.tstep_at_time(1106), (24,4))
        self.assertEqual(reshaped.tstep_at_time(1106, inclusive=False), (25,4))
        self.assertEqual(reshaped.tstep_at_time(1106, inclusive=False,
                                                      remask=True), Pair.MASKED)

        self.assertEqual(reshaped.tstep_range_at_time( 99.), ((0,0), (1,0)))
        self.assertEqual(reshaped.tstep_range_at_time( 99., remask=True)[0], Pair.MASKED)
        self.assertEqual(reshaped.tstep_range_at_time(100.), ((0,0), (1,1)))
        self.assertEqual(reshaped.tstep_range_at_time(106.), ((0,0), (1,1)))
        self.assertEqual(reshaped.tstep_range_at_time(110.), ((0,0), (1,2)))
        self.assertEqual(reshaped.tstep_range_at_time(139.999), ((0,3), (1,4)))
        self.assertRaises(ValueError, reshaped.tstep_range_at_time, 140)

        self.assertEqual(reshaped.tstep_range_at_time(1090), ((24,2), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1095.999), ((24,2), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1096), ((24,3), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1106), ((24,3), (25,4)))
        self.assertEqual(reshaped.tstep_range_at_time(1106, inclusive=False), ((24,3), (25,3)))
        self.assertEqual(reshaped.tstep_range_at_time(1106, inclusive=False,
                                                      remask=True)[0], Pair.MASKED)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
