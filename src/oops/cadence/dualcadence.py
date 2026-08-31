##########################################################################################
# oops/cadence/dualcadence.py
##########################################################################################

from polymath               import Scalar, Pair
from oops.cadence           import Cadence
from oops.cadence.metronome import Metronome


class DualCadence(Cadence):
    """A Cadence subclass in which time steps are defined by a pair of cadences."""

    def __init__(self, long, short):
        """Constructor for a DualCadence.

        Parameters:
            long (Cadence): The long or outer cadence. It defines the larger steps of the
                cadence, including the overall start time.
            short (Cadence): The short or inner cadence. It defines the time steps that
                break up the outer cadence, including the exposure time.

        Raises:
            ValueError: If either cadence is not 1-D.
        """

        self._long = long
        self._short = short.time_shift(-short.time[0])   # starts at time 0

        self.shape = self._long.shape + self._short.shape
        if len(self._long.shape) != 1 or len(self._short.shape) != 1:
            raise ValueError('long and short cadences must be 1-D')

        self.time = (self._long.time[0],
                     self._long.lasttime + self._short.time[1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self._long.lasttime + self._short.lasttime

        # self._short begins at time zero, so self._short.time[1] is the duration
        # spanned within each long time step. The cadence is continuous only if that
        # duration reaches the next long time step, and unique only if it does not
        # overlap the next long time step.
        self.is_continuous = (self._short.is_continuous and
                              self._short.time[1] >= self._long.max_tstride)

        self.is_unique = (self._short.is_unique and
                          self._short.time[1] <= self._long.min_tstride)

        self.min_tstride = self._short.min_tstride
        self.max_tstride = max(self._long.max_tstride - self._short.time[1],
                               self._short.max_tstride)

        self._max_long_tstep = self._long.shape[0] - 1

    def _refresh(self):
        """Update internals if self._long or self._short is Fittable."""
        self.time = (self._long.time[0],
                     self._long.lasttime + self._short.time[1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self._long.lasttime + self._short.lasttime

    def __getstate__(self):
        self.refresh()
        return (self._long, self._short)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

        self.time = (self._long.time[0],
                     self._long.lasttime + self._short.time[1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self._long.lasttime + self._short.lasttime

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        In this 2-D cadence, indexing beyond the dimensions of the cadence returns the
        time at the nearest edge of the cadence's shape.

        Parameters:
            tstep (Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar: Time in seconds TDB.
        """

        tstep = Pair.as_pair(tstep, recursive=derivs)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time = self._long.time_range_at_tstep(long_tstep, remask=remask,
                                                  inclusive=inclusive)[0]

        # Determine short time
        short_time = self._short.time_at_tstep(short_tstep, remask=remask,
                                              derivs=derivs,
                                              inclusive=inclusive)

        return long_time + short_time

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        In this 2-D cadence, indexing beyond the dimensions of the cadence returns the
        time range at the nearest edge.

        Parameters:
            tstep (Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index == shape) into the previous time step.

        Returns:
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the index
            values, in seconds TDB.
        """

        tstep = Pair.as_pair(tstep, recursive=False)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time0 = self._long.time_range_at_tstep(long_tstep, remask=remask,
                                                   inclusive=inclusive,
                                                   shift=shift)[0]

        # Determine short time range
        short_times = self._short.time_range_at_tstep(short_tstep,
                                                     remask=remask,
                                                     inclusive=inclusive,
                                                     shift=shift)

        return (long_time0 + short_times[0], long_time0 + short_times[1])

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
            Pair: Time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # Determine long tstep
        # We need remask=False because the end time of each long cadence is
        # ignored; remask=True might mask some times incorrectly.
        tstep0 = self._long.tstep_range_at_time(time, remask=False,
                                               inclusive=inclusive)[0]

        # Determine short tstep
        time0 = self._long.time_at_tstep(tstep0, remask=remask,
                                        inclusive=inclusive)
        tstep1 = self._short.tstep_at_time(time - time0, remask=remask,
                                          derivs=derivs,
                                          inclusive=inclusive)

        # Revise long time step above the time limits
        if inclusive:
            tstep0[time.vals > self.time[1]] = self.shape[0]
        else:
            tstep0[time.vals >= self.time[1]] = self.shape[0]

        return Pair.from_scalars(tstep0, tstep1)

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            tuple[Pair, Pair]: The range of time step indices active at the given `time`,
            as (first, last+1); the upper limit is excluded. Values are always within the
            allowed range for the cadence, regardless of any mask. If `time` is not
            sampled by the cadence, the range is empty, meaning that the second value
            equals the first.

        Raises:
            NotImplementedError: If the cadence is not unique, meaning that its time steps
                overlap in time.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Find integer tsteps at or below the time values, unmasked.
        # Times before the start time map to tstep0_min = 0;
        # Times during or after the last time step map to shape[0]-1.
        tstep0_min = self._long.tstep_range_at_time(time, remask=False,
                                                   inclusive=False)[0]
        tstep0_max = tstep0_min + 1

        # Unique case is MUCH easier
        if self.is_unique:

            # Determine short tstep range
            time0 = self._long.time_at_tstep(tstep0_min, remask=remask,
                                            inclusive=inclusive)

            # Note: exclude the last moment of each short cadence
            # We address the last moment of the cadence overall below
            (tstep1_min,
             tstep1_max) = self._short.tstep_range_at_time(time - time0,
                                                          remask=remask,
                                                          inclusive=False)

            # Time step ranges outside time limits are already zero-length

            # Handle the last moment of the cadence
            if inclusive:
                mask = (time.vals == self.time[1]) & time.antimask
                tstep1_min[mask] = self.shape[1] - 1    # this also unmasks
                tstep1_max[mask] = self.shape[1]

        else:
            raise NotImplementedError('tstep_range_at_time is not implemented '+
                                      'for a non-unique DualCadence')

        # This step merges the tstep1 mask over the incomplete tstep0 masks
        tstep_min = Pair.from_scalars(tstep0_min, tstep1_min)
        tstep_max = Pair.from_scalars(tstep0_max, tstep1_max)
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

        time = Scalar.as_scalar(time, recursive=False)

        # Easier case
        if self.is_continuous:
            if inclusive:
                return (time < self.time[0]) | (time > self.time[1])
            else:
                return (time < self.time[0]) | (time >= self.time[1])

        # Determine long tstep
        tstep0 = self._long.tstep_range_at_time(time, inclusive=inclusive)[0]

        # Test for short tstep
        time0 = self._long.time_at_tstep(tstep0, inclusive=inclusive)
        return self._short.time_is_outside(time - time0, inclusive=inclusive)

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            DualCadence: The time-shifted cadence.
        """

        return DualCadence(self._long.time_shift(secs), self._short)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For DualCadence, this is accomplished by forcing the short cadence to be
        continuous.

        Returns:
            DualCadence: The continuous cadence.

        Raises:
            ValueError: If the short cadence cannot be extended far enough to make this
                DualCadence continuous.
        """

        short = self._short.as_continuous()
        if short.time[1] >= self._long.max_tstride:
            return DualCadence(self._long, short)

        raise ValueError('short internal cadence cannot be extended to make ' +
                         'this DualCadence continuous')

    @staticmethod
    def for_array2d(samples, lines, tstart, texp, intersample_delay=0.,
                                                  interline_delay=None):
        """Alternative constructor for a DualCadence involving two Metronome classes, with
        streamlined input.

        Parameters:
            samples (int): Number of samples (along fast axis).
            lines (int): Number of lines (along slow axis).
            tstart (float): Start time of observation in TDB seconds.
            texp (float): Single-sample integration time in seconds.
            intersample_delay (float, optional): Deadtime in seconds between consecutive
                samples; default 0.
            interline_delay (float, optional): Deadtime in seconds between consecutive
                lines, i.e., the delay between the end of the last sample integration on
                one line and the start of the first sample integration on the next line.
                If not specified, the interline_delay matches the intersample_delay.

        Returns:
            DualCadence: The new cadence.
        """

        fast_cadence = Metronome(tstart, texp + intersample_delay, texp,
                                 samples)

        if interline_delay is None:
            interline_delay = intersample_delay

        long_texp = samples * texp + (samples-1) * intersample_delay
        long_stride = long_texp + interline_delay

        slow_cadence = Metronome(tstart, long_stride, long_texp, lines)

        return DualCadence(slow_cadence, fast_cadence)

##########################################################################################
