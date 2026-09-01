##########################################################################################
# oops/cadence/reshapedcadence.py
##########################################################################################

import numpy as np

from polymath     import Qube, Scalar, Pair, Vector
from oops.cadence import Cadence


class ReshapedCadence(Cadence):
    """A Cadence that has been reshaped.

    The time steps are defined by another cadence with a different shape. This can be
    used, for example, to convert a 1-D cadence into an N-D cadence.
    """

    def __init__(self, cadence, shape):
        """Constructor for a ReshapedCadence.

        Parameters:
            cadence (Cadence): The cadence to re-shape.
            shape (tuple): The new shape of the cadence.

        Raises:
            ValueError: If the new shape is incompatible with the size of the given
                cadence, or if it has more than two dimensions.
        """

        self._cadence = cadence
        self.shape = tuple(shape)
        self._rank = len(self.shape)
        self._size = int(np.prod(self.shape))

        if self._size != np.prod(self._cadence.shape):
            raise ValueError('ReshapedCadence size and shape are incompatible')

        if self._rank > 2:
            raise ValueError('%d-D cadences are not supported' % self._rank)

        self.time = self._cadence.time
        self.midtime = self._cadence.midtime
        self.lasttime = self._cadence.lasttime
        self.is_continuous = self._cadence.is_continuous
        self.is_unique = self._cadence.is_unique
        self.min_tstride = self._cadence.min_tstride
        self.max_tstride = self._cadence.max_tstride

        self._stride = np.cumprod((self.shape + (1,))[::-1])[-2::-1]
            # trust me, it works!

        self._old_shape = self._cadence.shape
        self._old_rank = len(self._cadence.shape)
        self._old_stride = np.cumprod((self._old_shape + (1,))[::-1])[-2::-1]

    def _refresh(self):
        """Update internals if self._cadence is Fittable."""
        self.time = self._cadence.time
        self.midtime = self._cadence.midtime
        self.lasttime = self._cadence.lasttime

    def __getstate__(self):
        self.refresh()
        return (self._cadence, self.shape)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    @staticmethod
    def _reshape_tstep(tstep, old_shape, old_stride, old_rank,
                              new_shape, new_stride, new_rank, size,
                              remask=False, derivs=False, inclusive=True):
        """Translate a cadence index from old shape to new."""

        # Convert old tstep to integer offset + fraction; remask for now
        if old_rank == 1:
            tstep = Scalar.as_scalar(tstep, recursive=derivs)
            index_1d = tstep.int(old_shape[0], remask=True, inclusive=inclusive,
                                 clip=True)
            remainder = tstep - index_1d
            frac = remainder.clip(0, 1, remask=True)
            index_1d = index_1d.vals
        else:
            tstep = Vector.as_vector(tstep, recursive=derivs)
            tstep_int = tstep.int(old_shape, remask=True, inclusive=inclusive, clip=True)
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
            indices = np.asarray(indices, dtype=np.float64)

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
                    # Note that the above works if new_deriv_vals has no shape, because
                    # indx is an Ellipsis, which is a valid index, and because the
                    # constructor converts a shapeless array value to a scalar.

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

    def _old_tstep_from_new(self, tstep, remask=False, derivs=False, inclusive=True):
        """Convert a tstep index for the old cadence to the new."""

        return ReshapedCadence._reshape_tstep(
                            tstep,
                            self.shape, self._stride, self._rank,
                            self._old_shape, self._old_stride, self._old_rank,
                            self._size,
                            remask=remask, derivs=derivs, inclusive=inclusive)

    def _new_tstep_from_old(self, tstep, remask=False, derivs=False, inclusive=True):
        """Convert a tstep index for the new cadence to the old."""

        return ReshapedCadence._reshape_tstep(tstep,
                            self._old_shape, self._old_stride, self._old_rank,
                            self.shape, self._stride, self._rank, self._size,
                            remask=remask, derivs=derivs, inclusive=inclusive)

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time at the nearest edge of the cadence's shape.

        Parameters:
            tstep (Scalar or Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar: Time in seconds TDB.
        """

        tstep = self._old_tstep_from_new(tstep, remask=remask, derivs=derivs,
                                         inclusive=inclusive)

        return self._cadence.time_at_tstep(tstep, remask=remask, derivs=derivs,
                                           inclusive=inclusive)

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time range at the nearest edge.

        Parameters:
            tstep (Scalar or Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index == shape) into the previous time step.

        Returns:
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the index
            values, in seconds TDB.
        """

        tstep = self._old_tstep_from_new(tstep, derivs=False, remask=remask,
                                         inclusive=inclusive)

        return self._cadence.time_range_at_tstep(tstep, remask=remask,
                                                 inclusive=inclusive, shift=shift)

    def tstep_at_time(self, time, *, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps via interpolation.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            derivs (bool, optional): True to include derivatives of time in the returned
                tstep.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar or Pair: Time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # Converting to 1-D or continuous cadences, this is fairly easy...
        if self._rank == 1 or self.is_continuous:
            tstep = self._cadence.tstep_at_time(time, remask=remask, derivs=derivs,
                                                inclusive=inclusive)
            tstep = self._new_tstep_from_old(tstep, remask=remask, derivs=derivs,
                                             inclusive=inclusive)

        # Otherwise...
        else:

            # Remove the time mask and remask, so tstep is masked if and only if
            # a time is out of range, including in a gap between discontinuous
            # time steps.
            tstep = self._cadence.tstep_at_time(time.without_mask(), remask=True,
                                                derivs=derivs, inclusive=inclusive)
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

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            tuple[Scalar or Pair, Scalar or Pair]: The range of time step indices active
            at the given `time`, as (first, last+1); the upper limit is excluded. Values
            are always within the allowed range for the cadence, regardless of any mask.
            If `time` is not sampled by the cadence, the range is empty, meaning that the
            second value equals the first.

        Raises:
            ValueError: If the reshaping would make the range of active time steps
                discontinuous.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Mask here; update the mask later if necessary
        (old_tstep_min,
         old_tstep_max) = self._cadence.tstep_range_at_time(time, remask=True,
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

            new_tstep_max = Pair(max_vals, new_tstep_min.mask)

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

                    raise ValueError('returned tstep range is discontinuous at '
                                     f'{timeval}: {minval}, {maxval}')

        # Make sure that the old tstep range was continuous
        if not self.is_unique and self._old_rank > 1:
            multiple_rows = np.any(old_tstep_min.vals[...,:-1] !=
                                   old_tstep_max.vals[...,:-1] - 1)
            incomplete_lines = ((old_tstep_min.vals[...,-1] != 0) |
                                (old_tstep_max.vals[...,-1] != self._old_shape[-1]))
            unmasked = old_tstep_min.antimask
            problems = multiple_rows & incomplete_lines & unmasked
            if np.any(problems):
                if np.isscalar(problems):
                    timeval = time.vals
                    minval = old_tstep_min
                    maxval = old_tstep_max
                else:
                    timeval = time[problems][0]
                    minval = old_tstep_min[problems][0]
                    maxval = old_tstep_max[problems][0]

                raise ValueError('input tstep range is discontinuous at ' +
                                 f'{timeval}: {minval}, {maxval}')

        # Restore the original mask if necessary
        if not remask:
            new_tstep_min = new_tstep_min.remask(time.mask)
            new_tstep_max = new_tstep_max.remask(time.mask)

        return (new_tstep_min, new_tstep_max)

    def time_is_outside(self, time, *, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Parameters:
            time (Scalar): Times in seconds TDB.
            inclusive (bool, optional): True to treat the end time of an interval as
                inside; False to treat it as outside. The start time of an interval is
                always treated as inside.

        Returns:
            Boolean: True where `time` is not sampled by the cadence.
        """

        return self._cadence.time_is_outside(time, inclusive=inclusive)

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            ReshapedCadence: The time-shifted cadence.
        """

        return ReshapedCadence(self._cadence.time_shift(secs), self.shape)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For ReshapedCadence, this is accomplished by making the underlying cadence
        continuous.

        Returns:
            ReshapedCadence: The continuous cadence.
        """

        return ReshapedCadence(self._cadence.as_continuous(), self.shape)

##########################################################################################
