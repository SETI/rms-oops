##########################################################################################
# oops/cadence/tdicadence.py
##########################################################################################

from polymath     import Scalar
from oops.cadence import Cadence


class TDICadence(Cadence):
    """A Cadence subclass defining the integration intervals of lines in a TDI ("Time
    Delay and Integration") camera. The tstep index matches the line index in the TDI
    detector.
    """

    def __init__(self, lines, tstart, tdi_texp, tdi_stages, tdi_sign=-1):
        """Constructor for a TDICadence.

        Parameters:
            lines (int): Lines in the detector. This corresponds to the number of time
                steps in the cadence.
            tstart (float): The start time of the observation in seconds TDB.
            tdi_texp (float): The interval in seconds from the start of one TDI step to
                the start of the next.
            tdi_stages (int): The number of TDI stages, 1 to the number of lines.
            tdi_sign (int, optional): +1 if pixel DNs are shifted in the positive
                direction along the 'ut' or 'vt' axis; -1 if DNs are shifted in the
                negative direction. Default is -1, suitable for JunoCam.

        Raises:
            ValueError: If `tdi_stages` is not between 1 and `lines`, inclusive.
        """

        # Save the input parameters
        self._lines = int(lines)
        self._tstart = float(tstart)
        self._tdi_texp = float(tdi_texp)
        self._tdi_stages = int(tdi_stages)
        self._tdi_sign = 1 if tdi_sign > 0 else -1

        if self._tdi_stages < 1 or self._tdi_stages > self._lines:
            raise ValueError('invalid TDICadence inputs: '
                             f'lines={lines}; tdi_stages={tdi_stages}')

        self._tdi_upward = (self._tdi_sign > 0)
        self._max_shifts = self._tdi_stages - 1
        self._max_line = self._lines - 1

        # Number of lines that are always active
        self._perm_lines = self._lines - self._max_shifts

        # Fill in the required attributes
        self.time = (self._tstart, self._tstart + self._tdi_texp * self._tdi_stages)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.time[1] - self._tdi_texp
        self.shape = (self._lines,)
        self.is_continuous = True
        self.is_unique = (self._tdi_stages == 1)
        self.min_tstride = 0.
        self.max_tstride = tdi_texp

        self._scalar_end_time = Scalar(self.time[1])

    def __getstate__(self):
        self.refresh()
        return (self._lines, self._tstart, self._tdi_texp, self._tdi_stages,
                self._tdi_sign)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    ######################################################################################
    # Methods unique to this class
    ######################################################################################

    def tdi_shifts_at_line(self, line, *, remask=False, inclusive=True, shift=True):
        """The number of TDI shifts at the given image line (or tstep).

        Parameters:
            line (Scalar): Line number, which is also the time step index.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it. If inclusive is False and remask is True,
                the end time is masked.
            shift (bool, optional): True to shift the end of the last time step (with
                index == shape) into the previous time step.

        Returns:
            Scalar: The number of TDI shifts at each line number.
        """

        line = Scalar.as_scalar(line, recursive=False)
        line = line.int(top=self._lines, remask=remask, inclusive=inclusive, shift=shift)

        if self._tdi_upward:
            shifts = line
        else:
            shifts = self._max_line - line

        return shifts.clip(0, self._max_shifts, remask=False)

    def tdi_shifts_after_time(self, time, *, remask=False, inclusive=True):
        """The number of TDI shifts remaining after the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it. If inclusive is False and remask is True,
                the end time is masked.

        Returns:
            Scalar: The number of TDI shifts that will occur after this time in the
            exposure.
        """

        time = Scalar.as_scalar(time, recursive=False)
        tstep = (time - self.time[0]) / self._tdi_texp
        tstep_int = tstep.int(top=self._tdi_stages, remask=remask, inclusive=inclusive)
        return (self._max_shifts - tstep_int).clip(0, self._max_shifts, remask=remask)

    ######################################################################################
    # Standard Cadence methods
    ######################################################################################

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
        tstep_int = tstep.int(top=self._lines, remask=remask, inclusive=inclusive,
                              clip=True)
        tstep_frac = (tstep - tstep_int).clip(0, 1, inclusive=inclusive, remask=False)

        (time_min, time_max) = self.time_range_at_tstep(tstep_int, remask=False)

        return time_min + tstep_frac * (time_max - time_min)

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        Every time step ends at the end time of the cadence; only the start times differ.

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

        stages = self.tdi_shifts_at_line(tstep, remask=remask, inclusive=inclusive,
                                         shift=shift) + 1

        time0 = self.time[1] - stages * self._tdi_texp
        time1 = Scalar.filled(time0.shape, self.time[1], mask=time0.mask)
        return (time0, time1)

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

        Raises:
            NotImplementedError: If the cadence has more than one TDI stage, in which case
                time values are not unique.
        """

        if self._tdi_stages > 1:
            raise NotImplementedError('TDICadence.tstep_at_time cannot be implemented; '
                                      'time values are not unique')

        time = Scalar.as_scalar(time, recursive=derivs)
        tstep = (time - self.time[0]) / self._tdi_texp
        return tstep.clip(0, 1, inclusive=inclusive, remask=remask)

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
        shifts = (time - self.time[0]) / self._tdi_texp

        # remask = True here; fix it below
        shifts = shifts.int(top=self._tdi_stages, remask=True, inclusive=inclusive,
                            clip=True)

        if self._tdi_upward:
            line_min = self._max_shifts - shifts
            line_max = Scalar.filled(shifts.shape, self._lines)
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

        return Cadence.time_is_outside(self, time, inclusive=inclusive)

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            TDICadence: The time-shifted cadence.
        """

        return TDICadence(self._lines, self._tstart + secs, self._tdi_texp,
                          self._tdi_stages, self._tdi_sign)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        A TDICadence is always continuous, so this returns the cadence itself.

        Returns:
            TDICadence: This cadence.
        """

        return self

##########################################################################################
