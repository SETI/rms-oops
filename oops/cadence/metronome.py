################################################################################
# oops/cadence/metronome.py: Metronome subclass of class Cadence
################################################################################

import numpy as np

from polymath     import Scalar, Qube
from oops.cadence import Cadence

class Metronome(Cadence):
    """A Cadence subclass where time steps occur at uniform intervals."""

    def __init__(self, tstart, tstride, texp, steps, clip=True):
        """Constructor for a Metronome.

        Input:
            tstart      the start time of the observation in seconds TDB.
            tstride     the interval in seconds from the start of one time step
                        to the start of the next.
            texp        the exposure time in seconds associated with each step.
                        This may be shorter than tstride due to readout times,
                        etc. It may also be longer.
            steps       the number of time steps.
            clip        if True (the default), times and index values are always
                        clipped into the valid range.
        """

        self.tstart = float(tstart)
        self.tstride = float(tstride)
        self.texp = float(texp)
        self.steps = int(steps)
        self.clip = bool(clip)

        if self.steps == 1:
            self.tstride = self.texp

        # Required attributes
        self.lasttime = self.tstart + self.tstride * (self.steps - 1)
        self.time = (self.tstart, self.lasttime + self.texp)
        self.midtime = (self.time[0] + self.time[1]) * 0.5
        self.shape = (self.steps,)
        self.is_continuous = (self.texp >= self.tstride)
        self.is_unique = (self.texp <= self.tstride)
        self.min_tstride = self.tstride
        self.max_tstride = self.tstride

        self._gapless = (self.texp == self.tstride)
        self._tscale = self.tstride / self.texp
        self._tspan = self.texp / self.tstride
        self._tspan1 = self._tspan - 1
        self._max_step = self.steps - 1

    def __getstate__(self):
        return (self.tstart, self.tstride, self.texp, self.steps,
                             self.clip)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)

        # One case is especially easy
        if not remask and not self.clip and self._gapless:
            return self.time[0] + self.tstride * tstep

        # Other cases
        tstep_int = tstep.int(top=self.steps, remask=remask,
                              inclusive=inclusive, clip=self.clip)
        tstep_frac = (tstep - tstep_int).clip(0, 1, remask=remask,
                                                    inclusive=False)
            # inclusive is False because the end moments of discontinuous time
            # steps are never included, except for the end of the final time
            # step, which is included when inclusive=True.

        # End moment might require special handling
        if inclusive and (remask or derivs):
            mask = (tstep == self.steps)
            tstep_frac[mask] = tstep[mask] - self._max_step
                # this sets the value to 1 but preserves derivatives

        return (self.time[0] + tstep_int * self.tstride
                             + tstep_frac * self.texp)

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

        tstep = Scalar.as_scalar(tstep, recursive=False)
        tstep_int = tstep.int(top=self.steps, remask=remask,
                              inclusive=inclusive, clip=self.clip)
        time_min = self.time[0] + tstep_int * self.tstride

        return (time_min, time_min + self.texp)

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps via interpolation.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of time in the returned
                        tstep.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Scalar of time step indices.
        """

        time = Scalar.as_scalar(time, recursive=derivs)
        tstep = (time - self.time[0]) / self.tstride

        if self._gapless:
            if self.clip:
                tstep = tstep.clip(0, self.steps, remask=remask,
                                   inclusive=inclusive)
            elif remask:
                tstep = tstep.mask_where_outside(0, self.steps, remask=True,
                                                 mask_endpoints=(False,
                                                                 not inclusive))

        elif self.is_unique:
            tstep_int = tstep.int(top=self.steps, remask=remask,
                                  inclusive=inclusive, clip=self.clip)
            tstep_diff = tstep - tstep_int
                # Regardless of self.clip, at the top...
                # If inclusive, tstep_int = self.steps-1 and tstep_diff = texp
                # Otherwise, tstep_int = self.steps and tstep_diff = 0.

            # If self.clip is True, then tstep_diff < 0. before the start time.
            # Otherwise, tstep_diff cannot be negative.
            if self.clip:
                tstep_diff[tstep_diff.vals < 0.] = Scalar(0., remask)

            # Don't let an interior fractional part match or exceed tspan, which
            # happens in the gaps between tsteps. However, if inclusive is True,
            # then the fractional part is allowed to equal tspan at the end
            # time.
            if inclusive:
                mask = ((tstep_diff.vals >= self._tspan)
                        & (time.vals != self.time[1]))
            else:
                mask = (tstep_diff.vals >= self._tspan)

            tstep_diff[mask] = Scalar(self._tspan, remask)

            # Now we can add the integer and fractional parts
            tstep = tstep_int + tstep_diff * self._tscale

        else:
            # Because time steps can overlap, avoid remask for now
            tstep_int = tstep.int(top=self.steps, remask=False,
                                  inclusive=False, clip=False)

            # Handle the last, extended time step
            is_last = Qube.is_inside(time.vals, self.lasttime, self.time[1],
                                     inclusive=inclusive)
            tstep_int[is_last] = self.steps - 1

            # Combine with fractional part
            tstep = tstep_int + (tstep - tstep_int) * self._tscale

            # Clip and remask necessary
            if self.clip:
                tstep = tstep.clip(0, self.steps,
                                   remask=remask, inclusive=inclusive)
            elif remask:
                endpoints = (False, not inclusive)
                tstep = tstep.mask_where_outside(0, self.steps,
                                                 mask_endpoints=endpoints)

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
            tstep_min   minimum Scalar time step containing the given time.
            tstep_max   maximum Scalar time step after the given time.

        Returned tstep_min will always be in the allowed range for the cadence,
        inclusive, regardless of masking. If the time is not inside the cadence,
        tstep_max == tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self.tstride

        # Set mask=True here; restore mask later if remask is False
        tstep_min = tstep.int(top=self.steps, remask=True,
                              inclusive=inclusive, clip=True)
        new_mask = tstep_min.mask       # Note: not a copy so modify cautiously

        # For discontinuous or gapless cases...
        if self.is_unique:
            tstep_max = tstep_min + 1

            # Expand mask for discontinuous cadences
            if not self.is_continuous:
                # Determine active time within each time step
                time_frac = (time.vals - self.time[0]
                                       - self.tstride * tstep_min.vals)

                # Mask times when integration is not happening
                if inclusive:       # extra care needed at end time
                    not_integrating = ((time_frac >= self.texp) &
                                       (time.vals != self.time[1]))
                else:
                    not_integrating = (time_frac >= self.texp)

                new_mask = Qube.or_(new_mask, not_integrating)

        else:
            # For overlapping cases...
            tstep_max = tstep_min + 1
            tstep_min = (tstep - self._tspan1).int(top=self.steps, remask=True,
                                                   inclusive=inclusive,
                                                   clip=True)
            # The new mask only applies if _both_ min and max are masked;
            # Otherwise, it is just a time near the beginning or end, and is
            # associated with fewer time steps, not no time steps.
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

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Masked time values return masked results.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        if self.is_continuous:
            return Cadence.time_is_outside(self, time, inclusive=inclusive)

        time = Scalar.as_scalar(time, recursive=False)
        time_mod = (time - self.time[0]) % self.tstride

        # Use TVL comparison to propagate the mask of time_mod
        if inclusive:
            return (time_mod.tvl_gt(self.texp) | time.tvl_lt(self.time[0])
                                               | time.tvl_gt(self.time[1]))
        else:
            return (time_mod.tvl_gt(self.texp) | time.tvl_lt(self.time[0])
                                               | time.tvl_ge(self.time[1]))

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return Metronome(self.tstart + secs,
                         self.tstride, self.texp, self.steps)

    #===========================================================================
    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous.

        For Metronome this is accomplished by forcing the exposure times to
        be equal to the stride.
        """

        return Metronome(self.tstart, self.tstride, self.tstride, self.steps)

    #===========================================================================
    def tstride_at_tstep(self, tstep, sign=1, remask=False):
        """The time interval(s) between the times of adjacent time steps.

        Input:
            tstep       a Scalar or Pair time step index, which need not be
                        integral.
            sign        +1 for the time interval to the next time step;
                        -1 for the time interval since the previous time step.
            remask      True to mask time tsteps that are out of range.

        Return:         a Scalar or Pair of strides in seconds.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)

        if remask:
            tstep = tstep.clip(0, self.steps, remask=remask)
            if np.any(tstep.mask):
                return Scalar.filled(tstep.shape, self.tstride, mask=tstep.mask)

        if np.shape(tstep.mask):
            return Scalar.filled(tstep.shape, self.tstride, mask=tstep.mask)

        return Scalar(self.tstride)

    #===========================================================================
    @staticmethod
    def for_array1d(steps, tstart, texp, interstep_delay=0.):
        """Alternative constructor.

        Input:
            steps               number of time steps.
            tstart              start time in seconds TDB.
            texp                exposure duration in second for each sample.
            interstep_delay     time delay in seconds between the end of one
                                integration and the beginning of the next, in
                                seconds. Default is 0.
        """

        return Metronome(tstart, texp + interstep_delay, texp, steps)

    #===========================================================================
    @staticmethod
    def for_array0d(tstart, texp):
        """Alternative constructor for a product with no time-axis.

        Input:
            tstart              start time in seconds TDB.
            texp                exposure duration in seconds.
        """

        return Metronome(tstart, texp, texp, 1)

################################################################################
