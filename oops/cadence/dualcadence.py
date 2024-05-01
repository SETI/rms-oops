################################################################################
# oops/cadence/dualcadence.py: DualCadence subclass of class Cadence
################################################################################

from polymath               import Scalar, Pair
from oops.cadence           import Cadence
from oops.cadence.metronome import Metronome

class DualCadence(Cadence):
    """A Cadence subclass in which time steps are defined by a pair of cadences.
    """

    #===========================================================================
    def __init__(self, long, short):
        """Constructor for a DualCadence.

        Input:
            long        the long or outer cadence. It defines the larger steps
                        of the cadence, including the overall start time.
            short       the short or inner cadence. It defines the time steps
                        that break up the outer cadence, including the exposure
                        time.
        """

        self.long = long
        self.short = short.time_shift(-short.time[0])   # starts at time 0

        self.shape = self.long.shape + self.short.shape
        if len(self.long.shape) != 1 or len(self.short.shape) != 1:
            raise ValueError('long and short cadences must be 1-D')

        self.time = (self.long.time[0],
                     self.long.lasttime + self.short.time[1])
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.lasttime = self.long.lasttime + self.short.lasttime

        self.is_continuous = (self.short.is_continuous and
                              self.time[0] >= self.long.max_tstride)

        self.is_unique = (self.short.is_unique and
                          self.short.time[0] <= self.long.min_tstride)

        self.min_tstride = self.short.min_tstride
        self.max_tstride = max(self.long.max_tstride - self.short.time[1],
                               self.short.max_tstride)

        self._max_long_tstep = self.long.shape[0] - 1

    def __getstate__(self):
        return (self.long, self.short)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Input:
            tstep       a Pair of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Pair.as_pair(tstep, recursive=derivs)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time = self.long.time_range_at_tstep(long_tstep, remask=remask,
                                                  inclusive=inclusive)[0]

        # Determine short time
        short_time = self.short.time_at_tstep(short_tstep, remask=remask,
                                              derivs=derivs,
                                              inclusive=inclusive)

        return long_time + short_time

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

        tstep = Pair.as_pair(tstep, recursive=False)
        (long_tstep, short_tstep) = tstep.to_scalars()

        # Determine long start time
        long_time0 = self.long.time_range_at_tstep(long_tstep, remask=remask,
                                                   inclusive=inclusive)[0]

        # Determine short time range
        short_times = self.short.time_range_at_tstep(short_tstep,
                                                     remask=remask,
                                                     inclusive=inclusive)

        return (long_time0 + short_times[0], long_time0 + short_times[1])

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Pair of time step index values.
        """

        time = Scalar.as_scalar(time, recursive=derivs)

        # Determine long tstep
        # We need remask=False because the end time of each long cadence is
        # ignored; remask=True might mask some times incorrectly.
        tstep0 = self.long.tstep_range_at_time(time, remask=False,
                                               inclusive=inclusive)[0]

        # Determine short tstep
        time0 = self.long.time_at_tstep(tstep0, remask=remask,
                                        inclusive=inclusive)
        tstep1 = self.short.tstep_at_time(time - time0, remask=remask,
                                          derivs=derivs,
                                          inclusive=inclusive)

        # Revise long time step above the time limits
        if inclusive:
            tstep0[time.vals > self.time[1]] = self.shape[0]
        else:
            tstep0[time.vals >= self.time[1]] = self.shape[0]

        return Pair.from_scalars(tstep0, tstep1)

    #===========================================================================
    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (tstep_min, tstep_max)
            tstep_min   minimum Pair time step containing the given time.
            tstep_max   maximum Pair time step containing the given time
                        (inclusive).

        All returned indices will be in the allowed range for the cadence,
        inclusive, regardless of mask. If the time is not inside the cadence,
        tstep_max < tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Find integer tsteps at or below the time values, unmasked.
        # Times before the start time map to tstep0_min = 0;
        # Times during or after the last time step map to shape[0]-1.
        tstep0_min = self.long.tstep_range_at_time(time, remask=False,
                                                   inclusive=False)[0]
        tstep0_max = tstep0_min + 1

        # Unique case is MUCH easier
        if self.is_unique:

            # Determine short tstep range
            time0 = self.long.time_at_tstep(tstep0_min, remask=remask,
                                            inclusive=inclusive)

            # Note: exclude the last moment of each short cadence
            # We address the last moment of the cadence overall below
            (tstep1_min,
             tstep1_max) = self.short.tstep_range_at_time(time - time0,
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

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Easier case
        if self.is_continuous:
            if inclusive:
                return (time < self.time[0]) | (time > self.time[1])
            else:
                return (time < self.time[0]) | (time >= self.time[1])

        # Determine long tstep
        tstep0 = self.long.tstep_range_at_time(time, inclusive=inclusive)[0]

        # Test for short tstep
        time0 = self.long.time_at_tstep(tstep0, inclusive=inclusive)
        return self.short.time_is_outside(time - time0, inclusive=inclusive)

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return DualCadence(self.long.time_shift(secs), self.short)

    #===========================================================================
    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous.

        For DualCadence, this is accomplished by forcing the stride of
        the short cadence to be continuous.
        """

        if self.time[0] >= self.long._max_tstride:
            return DualCadence(self.long, self.short.as_continuous())

        raise ValueError('short internal cadence cannot be extended to make ' +
                         'this DualCadence continuous')

    #===========================================================================
    @staticmethod
    def for_array2d(samples, lines, tstart, texp, intersample_delay=0.,
                                                  interline_delay=None):
        """Alternative constructor for a DualCadence involving two Metronome
        classes, with streamlined input.

        Input:
            samples             number of samples (along fast axis).
            lines               number of lines (along slow axis).
            tstart              start time of observation in TDB seconds.
            texp                single-sample integration time in seconds.
            intersample_delay   deadtime in seconds between consecutive samples;
                                default 0.
            interline_delay     deadtime in seconds between consecutive lines,
                                i.e., the delay between the end of the last
                                sample integration on one line and the start of
                                the first sample integration on the next line.
                                If not specified, the interline_delay is assumed
                                to match the intersample_delay.
        """

        fast_cadence = Metronome(tstart, texp + intersample_delay, texp,
                                 samples)

        if interline_delay is None:
            interline_delay = intersample_delay

        long_texp = samples * texp + (samples-1) * intersample_delay
        long_stride = long_texp + interline_delay

        slow_cadence = Metronome(tstart, long_stride, long_texp, lines)

        return DualCadence(slow_cadence, fast_cadence)

################################################################################
