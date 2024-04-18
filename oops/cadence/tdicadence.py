################################################################################
# oops/cadence/tdicadence.py: TDICadence subclass of class Cadence
################################################################################

from polymath     import Scalar
from oops.cadence import Cadence

class TDICadence(Cadence):
    """A Cadence subclass defining the integration intervals of lines in a TDI
    ("Time Delay and Integration") camera. The tstep index matches the line
    index in the TDI detector.
    """

    def __init__(self, lines, tstart, tdi_texp, tdi_stages, tdi_sign=-1):
        """Constructor for a TDICadence.

        Input:
            lines       the number of lines in the detector. This corresponds to
                        the number of time steps in the cadence.
            tstart      the start time of the observation in seconds TDB.
            tdi_texp    the interval in seconds from the start of one TDI step
                        to the start of the next.
            tdi_stages  the number of TDI time steps, 1 to number of lines.
            tdi_sign    +1 if pixel DNs are shifted in the positive direction
                        along the 'ut' or 'vt' axis; -1 if DNs are shifted in
                        the negative direction. Default is -1, suitable for
                        JunoCam.
        """

        # Save the input parameters
        self.lines = int(lines)
        self.tstart = float(tstart)
        self.tdi_texp = float(tdi_texp)
        self.tdi_stages = int(tdi_stages)
        self.tdi_sign = 1 if tdi_sign > 0 else -1

        if self.tdi_stages < 1 or self.tdi_stages > self.lines:
            raise ValueError('invalid TDICadence inputs: ' +
                             'lines=%d; tdi_stages=%d' % (lines, tdi_stages))

        self._tdi_upward = (self.tdi_sign > 0)
        self._max_shifts = self.tdi_stages - 1
        self._max_line = self.lines - 1

        # Number of lines that are always active
        self._perm_lines = self.lines - self._max_shifts

        # Fill in the required attributes
        self.time = (self.tstart, self.tstart + self.tdi_texp * self.tdi_stages)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.time[-1] - self.tdi_texp
        self.shape = (self.lines,)
        self.is_continuous = True
        self.is_unique = (self.tdi_stages == 1)
        self.min_tstride = 0.
        self.max_tstride = tdi_texp

        self._scalar_end_time = Scalar(self.time[1])

    def __getstate__(self):
        return (self.lines, self.tstart, self.tdi_texp, self.tdi_stages,
                self.tdi_sign)

    def __setstate__(self, state):
        self.__init__(*state)

    ############################################################################
    # Methods unique to this class
    ############################################################################

    def tdi_shifts_at_line(self, line, remask=False, inclusive=True):
        """The number of TDI shifts at the given image line (or tstep).

        Input:
            line        a Scalar line number.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as inside the
                        cadence. If inclusive is False and remask is True, the
                        end time will be masked.

        Return:         an integer Scalar defining the number of TDI shifts at
                        this line number.
        """

        line = Scalar.as_scalar(line, recursive=False)
        line = line.int(top=self.lines, remask=remask, inclusive=inclusive)

        if self._tdi_upward:
            shifts = line
        else:
            shifts = self._max_line - line

        return shifts.clip(0, self._max_shifts, remask=False)

    #===========================================================================
    def tdi_shifts_after_time(self, time, remask=False, inclusive=True):
        """The number of TDI shifts at the given time.

        Input:
            time        Scalar of optional absolute time in seconds.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as inside the
                        cadence. If inclusive is False and remask is True, the
                        end time will be masked.

        Return:         an integer Scalar defining the number of TDI shifts that
                        will occur after this time in the exposure.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self.tdi_texp
        tstep_int = tstep.int(top=self.tdi_stages,
                              remask=remask, inclusive=inclusive)
        return (self._max_shifts - tstep_int).clip(0, self.tdi_stages,
                                                      remask=remask)

    ############################################################################
    # Standard Cadence methods
    ############################################################################

    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

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
        tstep_int = tstep.int(top=self.lines, remask=remask,
                              inclusive=inclusive, clip=True)
        tstep_frac = (tstep - tstep_int).clip(0, 1, inclusive=inclusive,
                                                    remask=False)

        (time_min,
         time_max) = self.time_range_at_tstep(tstep_int, remask=False)

        return time_min + tstep_frac * (time_max - time_min)

    #===========================================================================
    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Input:
            tstep       a Scalar of time step index values.
            remask      True to mask values outside the time limits.
            inclusive   True to treat the end time of the cadence as part of the
                        cadence; False to exclude it.

        Return:         (time_min, time_max)
            time_min    a Scalar defining the minimum time associated with the
                        index. It is given in seconds TDB.
            time_max    a Scalar defining the maximum time value.
        """

        stages = self.tdi_shifts_at_line(tstep, remask=remask,
                                                inclusive=inclusive) + 1

        time0 = self.time[1] - stages * self.tdi_texp
        time1 = Scalar.filled(time0.shape, self.time[1], mask=time0.mask)
        return (time0, time1)

    #===========================================================================
    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Input:
            time        a Scalar of times in seconds TDB.
            remask      True to mask time values not sampled within the cadence.
            derivs      True to include derivatives of tstep in the returned
                        time.
            inclusive   True to treat the end time of an interval as inside the
                        cadence; False to treat it as outside. The start time of
                        an interval is always treated as inside.

        Return:         a Scalar of time step index values.
        """

        if self.tdi_stages > 1:
            raise NotImplementedError('TDICadence.tstep_at_time cannot be ' +
                                      'implemented; time values are not unique')

        time = Scalar.as_scalar(time, recursive=derivs)
        tstep = (time - self.time[0]) / self.tdi_texp
        return tstep.clip(0, 1, inclusive=inclusive, remask=remask)

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
            tstep_max   minimum Scalar time step after the given time.

        Returned tstep_min will always be in the allowed range for the cadence,
        inclusive, regardless of masking. If the time is not inside the cadence,
        tstep_max == tstep_min.
        """

        time = Scalar.as_scalar(time, recursive=False)
        shifts = (time - self.time[0]) / self.tdi_texp

        # remask = True here; fix it below
        shifts = shifts.int(top=self.tdi_stages, remask=True,
                            inclusive=inclusive, clip=True)

        if self._tdi_upward:
            line_min = self._max_shifts - shifts
            line_max = Scalar.filled(shifts.shape, self.lines)
            line_max[shifts.mask] = line_min[shifts.mask]
        else:
            line_min = Scalar.zeros(shifts.shape, dtype='int', mask=shifts.mask)
            line_max = self._perm_lines + shifts
            line_max[shifts.mask] = line_min[shifts.mask]

        if remask:
            line_min = line_min.remask(shifts.mask)
            line_max = line_max.remask(shifts.mask)
        else:
            line_min = line_min.remask(time.mask)
            line_max = line_max.remask(time.mask)

        return (line_min, line_max)

    #===========================================================================
    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of time(s) that fall outside the cadence.

        Input:
            time        a Scalar of times in seconds TDB.
            inclusive   True to treat the end time of an interval as inside;
                        False to treat it as outside. The start time of an
                        interval is always treated as inside.

        Return:         a Boolean array indicating which time values are not
                        sampled by the cadence.
        """

        return Cadence.time_is_outside(self, time, inclusive)

    #===========================================================================
    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given
        amount.

        Input:
            secs        the number of seconds to shift the time later.
        """

        return TDICadence(self.tstart + secs, self.tdi_texp, self.tdi_stages,
                          self.tdi_sign, self.lines)

    #===========================================================================
    def as_continuous(self):
        """A shallow copy of this cadence, forced to be continuous."""

        return self

################################################################################
