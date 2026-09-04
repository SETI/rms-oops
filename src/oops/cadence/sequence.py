##########################################################################################
# oops/cadence/sequence.py
##########################################################################################

import numpy as np

from polymath     import Boolean, Scalar, Qube
from oops.cadence import Cadence


class Sequence(Cadence):
    """Cadence subclass in which time steps are defined by a list."""

    def __init__(self, tlist, texp):
        """Constructor for a Sequence.

        Parameters:
            tlist (Scalar, list, or numpy.ndarray): The start times of the time steps, in
                seconds TDB.
            texp (float, list, or numpy.ndarray): The exposure time in seconds associated
                with each step. This can be shorter than the time interval due to readout
                times, etc. It could also potentially be longer. The value can be:

                * a positive constant, indicating that exposure times are fixed;
                * a list or 1-D array, listing the exposure time associated with each time
                  step;
                * zero, indicating that each exposure lasts up to the start of the next
                  time step. In this case, the last tabulated time is the end time of the
                  previous exposure rather than the start of a final time step, so the
                  number of time steps is len(tlist)-1 rather than len(tlist).

        Raises:
            ValueError: If `tlist` is not 1-D, if `tlist` or `texp` is masked, if the
                shapes of `tlist` and `texp` do not match, or if any exposure time is not
                positive.
        """

        # Work with Numpy arrays initially
        if isinstance(tlist, Scalar):
            if np.any(tlist.mask):
                raise ValueError('Sequence tlist input must be unmasked')
            tlist = tlist.vals

        if isinstance(texp, Scalar):
            if np.any(texp.mask):
                raise ValueError('Sequence texp input must be unmasked')
            texp = texp.vals

        tlist = np.asarray(tlist, dtype=np.float64)
        if np.ndim(tlist) != 1 or tlist.size <= 1:
            raise ValueError('Sequence tlist must be 1-D')

        tstrides = np.diff(tlist)

        self._state_texp = texp

        # Interpret texp
        if np.shape(texp):          # texp is an array
            texp = np.asarray(texp, dtype=np.float64)
            if texp.shape != tlist.shape:
                raise ValueError('Shape mismatch between texp and tlist')
            if np.any(texp <= 0.):
                raise ValueError('All texp values must be positive')

            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = np.all(texp[:-1] >= tstrides)
            self.is_unique = np.all(texp[:-1] <= tstrides)

            tstop = tlist + texp
            self._tstop_is_ordered = bool(np.all(np.diff(tstop) > 0.))

        elif texp:                  # texp is a nonzero constant
            if (texp <= 0.):
                raise ValueError('All texp values must be positive')
            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = (texp >= self.max_tstride)
            self.is_unique = (texp <= self.min_tstride)

            # Create a filled array in place of the single value
            saved_texp = texp
            texp = np.empty(tlist.shape)
            texp.fill(saved_texp)

            tstop = tlist + texp
            self._tstop_is_ordered = True

        else:                       # use diffs to define texp
            texp = tstrides
            tstop = tlist[1:]
            tlist = tlist[:-1]      # last time is not a time step
            if np.any(texp <= 0.):
                raise ValueError('Sequence tlist inputs must be monotonic')

            tstrides = tstrides[:-1]
            self.min_tstride = np.min(tstrides)
            self.max_tstride = np.max(tstrides)
            self.is_continuous = True
            self.is_unique = True
            self._tstop_is_ordered = True

        # Convert back to Scalar and save
        # as_readonly() ensures that these inputs cannot be modified by
        # something external to the object.
        self._tlist  = Scalar(tlist).as_readonly()
        self._texp   = Scalar(texp).as_readonly()
        self._tstop = Scalar(tstop).as_readonly()

        self._steps = self._tlist.size
        self._max_tstep = self._steps - 1

        # Used for the inverse conversion
        self._interp_y = np.arange(self._steps, dtype='float')
        self._is_gapless = self.is_continuous and self.is_unique

        # Fill in required attributes
        self.lasttime = self._tlist.vals[-1]
        self.time = (self._tlist.vals[0], self._tlist.vals[-1] + self._texp.vals[-1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = self._tlist.shape

        return

    def __getstate__(self):
        self.refresh()
        return (self._tlist, self._state_texp)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        Parameters:
            tstep (Scalar): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar: Time in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)
        tstep_int = tstep.int(top=self._steps, remask=remask, clip=True,
                              inclusive=inclusive)
        tstep_frac = (tstep - tstep_int).clip(0, 1, remask=remask, inclusive=inclusive)

        time = (self._tlist[tstep_int.vals] + tstep_frac * self._texp[tstep_int.vals])
        return time

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        Parameters:
            tstep (Scalar): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index == shape) into the previous time step.

        Returns:
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the index
            values, in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)
        tstep_int = tstep.int(top=self._steps, remask=remask, clip=True,
                              inclusive=inclusive, shift=shift)

        time_min = Scalar(self._tlist[tstep_int.vals], tstep_int.mask)

        return (time_min, time_min + self._texp[tstep_int.vals])

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
            Scalar: Time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # np.interp converts each time to a float whose integer part is the
        # index of the time step at or below this time. Times outside the valid
        # range get mapped to the nearest valid index. As a result, any time
        # before the start time gets mapped to 0 and any time during or after
        # the last time step returns the last index, self._steps-1.
        #
        # Note that, if the Sequence integration times overlap and therefore
        # tstep_at_time does not have a unique solution, this will return the
        # last tstep that contains the time, which is probably what we want.

        interp = np.interp(time.vals, self._tlist.vals, self._interp_y)
        tstep_int = interp.astype('int')

        # tstep_frac is 0 at the beginning of each integration and 1 at the
        # end. It is negative before the first time step and > 1 after the end
        # of the last. We clip it (0 inclusive,1 exclusive) before adding it
        # back to the integer part.

        tstep_frac_unclipped = ((time - self._tlist[tstep_int])
                                / self._texp[tstep_int])
        tstep_frac_clipped = tstep_frac_unclipped.clip(0, 1, remask=remask,
                                                       inclusive=False)

        tstep = tstep_int + tstep_frac_clipped

        # The end time might require special handling, because it should be
        # unmasked if inclusive=True, whereas the end times of intermediate
        # time steps are not included.

        if inclusive:
            mask = Boolean.as_boolean(time == self.time[1])
            if mask.any():
                tstep[mask] = (Scalar.as_scalar(tstep_int)[mask]
                               + tstep_frac_unclipped[mask])

        return tstep

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it. If the cadence is not continuous, this also
                defines whether the end moment of each individual interval is included in
                that interval.

        Returns:
            tuple[Scalar, Scalar]: The range of time step indices active at the given
            `time`, as (first, last+1); the upper limit is excluded. Values are always
            within the allowed range for the cadence, regardless of any mask. If `time` is
            not sampled by the cadence, the range is empty, meaning that the second value
            equals the first.

        Raises:
            RuntimeError: If the stop times of the sequence are not strictly ordered.
        """

        if not self._tstop_is_ordered:
            raise RuntimeError('tstep_range_at_time failure in Sequence; '
                               'stop times are not strictly ordered')

        time = Scalar.as_scalar(time, recursive=False)

        # Locate the first stop time before and the last start time after
        tstep0 = np.interp(time.vals, self._tstop.vals, self._interp_y)
        tstep_min = Scalar(tstep0.astype('int'))        # last stop <= time

        temp_mask = (time.vals >= self._tstop[0]) & (time.vals < self.time[1])
        tstep_min[temp_mask] += 1                       # first stop > time

        tstep1 = np.interp(time.vals, self._tlist.vals, self._interp_y)
        tstep_max = Scalar(tstep1.astype('int')) + 1    # last start <= time + 1

        # Identify points outside the range for adjustment and masking
        # For all points outside range, tstep_max == tstep_min.
        # This also applies to times between time steps for discontinuous
        # cadences.
        if inclusive:
            mask = (time.vals < self.time[0]) | (time.vals > self.time[1])
            if not self.is_continuous:
                k = (tstep1.astype('int') if isinstance(tstep1, np.ndarray)
                                          else int(tstep1))
                mask |= ((time.vals - self._tlist.vals[k] >= self._texp.vals[k])
                         & (time.vals < self.time[1]))
        else:
            mask = (time.vals < self.time[0]) | (time.vals >= self.time[1])
            if not self.is_continuous:
                k = (tstep1.astype('int') if isinstance(tstep1, np.ndarray)
                                          else int(tstep1))
                mask |= (time.vals - self._tlist.vals[k] >= self._texp.vals[k])

        tstep_max[mask] = tstep_min[mask]

        # Update the mask
        if remask:
            if np.any(mask):
                new_mask = Qube.or_(time.mask, mask)
            else:
                new_mask = time.mask

            tstep_min = tstep_min.remask(new_mask)
            tstep_max = tstep_max.remask(new_mask)

        else:
            tstep_min = tstep_min.remask(time.mask)
            tstep_max = tstep_max.remask(time.mask)

        return (tstep_min, tstep_max)

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

        if self.is_continuous:
            return Cadence.time_is_outside(self, time, inclusive=inclusive)

        # See tstep_at_time above for explanation...
        time = Scalar.as_scalar(time, recursive=False)
        interp = np.interp(time.vals, self._tlist.vals, self._interp_y)

        # Convert to int, carefully...
        if np.isscalar(interp):
            tstep_int = int(interp)
        else:
            tstep_int = interp.astype('int')

        # Compare times, using TVL comparisons to retain the mask on time_diff
        time_diff = time - self._tlist.vals[tstep_int]
        if inclusive:
            is_outside = time_diff.tvl_lt(0.) | time_diff.tvl_gt(self._texp[tstep_int])
        else:
            is_outside = time_diff.tvl_lt(0.) | time_diff.tvl_ge(self._texp[tstep_int])

        return is_outside

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            Sequence: The time-shifted cadence.
        """

        return Sequence(self._tlist + secs, self._texp)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For Sequence, this is accomplished by forcing the exposure time of each step to be
        greater than or equal to its stride.

        Returns:
            Sequence: The continuous cadence.
        """

        if self.is_continuous:
            return self

        texp = np.empty(self._tlist.shape)
        texp[:-1] = np.maximum(self._texp.vals[:-1], np.diff(self._tlist.vals))
        texp[ -1] = self._texp[-1].vals

        result = Sequence(self._tlist, texp)
        result.is_continuous = True  # forced, in case of roundoff error in texp
        return result

##########################################################################################
