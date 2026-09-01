##########################################################################################
# oops/cadence/metronome.py
##########################################################################################

import numpy as np

from polymath     import Scalar, Qube
from oops.cadence import Cadence


class Metronome(Cadence):
    """A Cadence subclass where time steps occur at uniform intervals."""

    def __init__(self, tstart, tstride, texp, steps, *, clip=True):
        """Constructor for a Metronome.

        Parameters:
            tstart (float): The start time of the observation in seconds TDB.
            tstride (float): The interval in seconds from the start of one time step to
                the start of the next.
            texp (float): The exposure time in seconds associated with each step. This may
                be shorter than tstride due to readout times, etc. It may also be longer.
            steps (int): The number of time steps.
            clip (bool, optional): If True (the default), times and index values are
                always clipped into the valid range.
        """

        self._tstart = float(tstart)
        self._tstride = float(tstride)
        self._texp = float(texp)
        self._steps = int(steps)
        self._clip = bool(clip)

        if self._steps == 1:
            self._tstride = self._texp

        # Required attributes
        self.lasttime = self._tstart + self._tstride * (self._steps - 1)
        self.time = (self._tstart, self.lasttime + self._texp)
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = (self._steps,)
        self.is_continuous = (self._texp >= self._tstride)
        self.is_unique = (self._texp <= self._tstride)
        self.min_tstride = self._tstride
        self.max_tstride = self._tstride

        self._gapless = (self._texp == self._tstride)
        self._tscale = self._tstride / self._texp
        self._tspan = self._texp / self._tstride
        self._tspan1 = self._tspan - 1
        self._max_step = self._steps - 1

    def __getstate__(self):
        self.refresh()
        return (self._tstart, self._tstride, self._texp, self._steps, self._clip)

    def __setstate__(self, state):
        self.__init__(*state[:-1], clip=state[-1])
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

        # One case is especially easy
        if not remask and not self._clip and self._gapless:
            return self.time[0] + self._tstride * tstep

        # Other cases
        tstep_int = tstep.int(top=self._steps, remask=remask, inclusive=inclusive,
                              clip=self._clip)
        tstep_frac = (tstep - tstep_int).clip(0, 1, remask=remask, inclusive=False)
            # inclusive is False because the end moments of discontinuous time
            # steps are never included, except for the end of the final time
            # step, which is included when inclusive=True.

        # End moment might require special handling
        if inclusive and (remask or derivs):
            mask = (tstep == self._steps)
            tstep_frac[mask] = tstep[mask] - self._max_step
                # this sets the value to 1 but preserves derivatives

        return self.time[0] + tstep_int * self._tstride + tstep_frac * self._texp

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
        tstep_int = tstep.int(top=self._steps, remask=remask,
                              inclusive=inclusive, clip=self._clip, shift=shift)
        time_min = self.time[0] + tstep_int * self._tstride

        return (time_min, time_min + self._texp)

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
        tstep = (time - self.time[0]) / self._tstride

        if self._gapless:
            if self._clip:
                tstep = tstep.clip(0, self._steps, remask=remask, inclusive=inclusive)
            elif remask:
                tstep = tstep.mask_where_outside(0, self._steps, remask=True,
                                                 mask_endpoints=(False, not inclusive))

        elif self.is_unique:
            tstep_int = tstep.int(top=self._steps, remask=remask, inclusive=inclusive,
                                  clip=self._clip)
            tstep_diff = tstep - tstep_int
                # Regardless of self._clip, at the top...
                # If inclusive, tstep_int = self._steps-1 and tstep_diff = texp
                # Otherwise, tstep_int = self._steps and tstep_diff = 0.

            # If self._clip is True, then tstep_diff < 0. before the start time.
            # Otherwise, tstep_diff cannot be negative.
            if self._clip:
                tstep_diff[tstep_diff.vals < 0.] = Scalar(0., remask)

            # Don't let an interior fractional part match or exceed tspan, which happens
            # in the gaps between tsteps. However, if inclusive is True, then the
            # fractional part is allowed to equal tspan at the end time.
            if inclusive:
                mask = (tstep_diff.vals >= self._tspan) & (time.vals != self.time[1])
            else:
                mask = (tstep_diff.vals >= self._tspan)

            tstep_diff[mask] = Scalar(self._tspan, remask)

            # Now we can add the integer and fractional parts
            tstep = tstep_int + tstep_diff * self._tscale

        else:
            # Because time steps can overlap, avoid remask for now
            tstep_int = tstep.int(top=self._steps, remask=False, inclusive=False,
                                  clip=False)

            # Handle the last, extended time step
            is_last = Qube.is_inside(time.vals, self.lasttime, self.time[1],
                                     inclusive=inclusive)
            tstep_int[is_last] = self._steps - 1

            # Combine with fractional part
            tstep = tstep_int + (tstep - tstep_int) * self._tscale

            # Clip and remask necessary
            if self._clip:
                tstep = tstep.clip(0, self._steps, remask=remask, inclusive=inclusive)
            elif remask:
                endpoints = (False, not inclusive)
                tstep = tstep.mask_where_outside(0, self._steps, mask_endpoints=endpoints)

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
            tuple[Scalar, Scalar]: The range of time step indices active at the given
            `time`, as (first, last+1); the upper limit is excluded. Values are always
            within the allowed range for the cadence, regardless of any mask. If `time` is
            not sampled by the cadence, the range is empty, meaning that the second value
            equals the first.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self._tstride

        # Set mask=True here; restore mask later if remask is False
        tstep_min = tstep.int(top=self._steps, remask=True, inclusive=inclusive,
                              clip=True)
        new_mask = tstep_min.mask       # Note: not a copy so modify cautiously

        # For discontinuous or gapless cases...
        if self.is_unique:
            tstep_max = tstep_min + 1

            # Expand mask for discontinuous cadences
            if not self.is_continuous:
                # Determine active time within each time step
                time_frac = time.vals - self.time[0] - self._tstride * tstep_min.vals

                # Mask times when integration is not happening
                if inclusive:       # extra care needed at end time
                    not_integrating = ((time_frac >= self._texp) &
                                       (time.vals != self.time[1]))
                else:
                    not_integrating = (time_frac >= self._texp)

                new_mask = Qube.or_(new_mask, not_integrating)

        else:
            # For overlapping cases...
            tstep_max = tstep_min + 1
            tstep_min = (tstep - self._tspan1).int(top=self._steps, remask=True,
                                                   inclusive=inclusive, clip=True)
            # The new mask only applies if _both_ min and max are masked;
            # Otherwise, it is just a time near the beginning or end, and is associated
            # with fewer time steps, not no time steps.
            new_mask = Qube.and_(new_mask, tstep_min.mask)

        # Masked tstep ranges must have zero length
        tstep_max[new_mask] = tstep_min[new_mask]

        # Make sure both endpoints share a common mask
        if remask:
            tstep_min = tstep_min.remask(new_mask)
            tstep_max = tstep_max.remask(new_mask)
        else:
            # Without remasking, revert to the original mask
            tstep_min = tstep_min.remask(time.mask)
            tstep_max = tstep_max.remask(time.mask)

        return (tstep_min, tstep_max)

    def time_is_outside(self, time, *, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Masked time values return masked results.

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

        time = Scalar.as_scalar(time, recursive=False)
        time_mod = (time - self.time[0]) % self._tstride

        # Use TVL comparison to propagate the mask of time_mod
        if inclusive:
            return (time_mod.tvl_gt(self._texp) | time.tvl_lt(self.time[0])
                                                | time.tvl_gt(self.time[1]))
        else:
            return (time_mod.tvl_gt(self._texp) | time.tvl_lt(self.time[0])
                                                | time.tvl_ge(self.time[1]))

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            Metronome: The time-shifted cadence.
        """

        return Metronome(self._tstart + secs, self._tstride, self._texp, self._steps,
                         clip=self._clip)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For Metronome, this is accomplished by forcing the exposure time to equal the
        stride.

        Returns:
            Metronome: The continuous cadence.
        """

        return Metronome(self._tstart, self._tstride, self._tstride, self._steps,
                         clip=self._clip)

    def tstride_at_tstep(self, tstep, sign=1, *, remask=False):
        """The time interval(s) between the times of adjacent time steps.

        Parameters:
            tstep (Scalar): Time step index values.
            sign (int, optional): +1 for the time interval to the next time step; -1 for
                the time interval since the previous time step.
            remask (bool, optional): True to mask tsteps that are out of range.

        Returns:
            Scalar: Strides in seconds.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)

        if remask:
            tstep = tstep.clip(0, self._steps, remask=remask)
            if np.any(tstep.mask):
                return Scalar.filled(tstep.shape, self._tstride, mask=tstep.mask)

        if np.shape(tstep.mask):
            return Scalar.filled(tstep.shape, self._tstride, mask=tstep.mask)

        return Scalar(self._tstride)

    @staticmethod
    def for_array1d(steps, tstart, texp, interstep_delay=0.):
        """Alternative constructor.

        Parameters:
            steps (int): Number of time steps.
            tstart (float): Start time in seconds TDB.
            texp (float): Exposure duration in seconds for each sample.
            interstep_delay (float, optional): Time delay in seconds between the end of
                one integration and the beginning of the next. Default is 0.

        Returns:
            Metronome: The new cadence.
        """

        return Metronome(tstart, texp + interstep_delay, texp, steps)

##########################################################################################
