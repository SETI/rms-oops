##########################################################################################
# oops/cadence/reversedcadence.py
##########################################################################################

from polymath     import Scalar
from oops.cadence import Cadence


class ReversedCadence(Cadence):
    """A 1-D Cadence made by reversing the index order of a given Cadence.

    This is needed for cases where the times of pixels along an axis are in decreasing
    order as the data index increases.
    """

    def __init__(self, cadence, axis=0):
        """Constructor for a ReversedCadence.

        Parameters:
            cadence (Cadence): The cadence to reverse.
            axis (int, optional): The axis to reverse. Only axis 0 is supported.

        Raises:
            ValueError: If the given cadence is not 1-D, or if `axis` is not 0.
        """

        self._cadence = cadence
        if len(self._cadence.shape) != 1:
            raise ValueError('ReversedCadence must be based on a 1-D cadence')

        self._axis = int(axis)
        if self._axis != 0:
            raise ValueError(f'ReversedCadence axis must be 0, not {axis}')

        # Required attributes
        self.shape         = self._cadence.shape
        self.lasttime      = self._cadence.lasttime
        self.time          = self._cadence.time
        self.midtime       = self._cadence.midtime
        self.is_continuous = self._cadence.is_continuous
        self.is_unique     = self._cadence.is_unique
        self.min_tstride   = self._cadence.min_tstride
        self.max_tstride   = self._cadence.max_tstride

        # Used internally
        self._steps = self._cadence.shape[0]
        self._max_step = self._steps - 1

        # Beginning of new first time step; end of new last time step
        self._first_time = self._cadence.time_range_at_tstep(self._max_step)[0]
        self._last_time  = self._cadence.time_range_at_tstep(0)[1]

    def _refresh(self):
        """Update internals if self._cadence is Fittable."""
        self.time = self._cadence.time
        self.midtime = self._cadence.midtime
        self.lasttime = self._cadence.lasttime
        self._first_time = self._cadence.time_range_at_tstep(self._max_step)[0]
        self._last_time  = self._cadence.time_range_at_tstep(0)[1]

    def __getstate__(self):
        self.refresh()
        return (self._cadence, self._axis)

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

        # Reverse the order of the indices, but allow the fractional part to
        # increase within each time step.

        tstep_int = tstep.int(self._steps, remask=remask, inclusive=inclusive, shift=True)
            # Note: Because shift=True, the end of the last time step will map
            # into the first time step, yielding tstep_frac = 1 below,
            # regardless of whether it is to be included.

        reversed_tstep = self._max_step - tstep_int
        (time0, time1) = self._cadence.time_range_at_tstep(reversed_tstep, remask=False,
                                                           inclusive=False)
            # inclusive=False above because reversed_tstep == self._steps where
            # tstep_int == -1, which must be excluded. remask=False because the
            # input is already properly masked.

        tstep_frac = tstep - tstep_int
        time = time0 + tstep_frac * (time1 - time0)

        # Force out-of range tsteps to the start or end time
        time[tstep_int.vals < 0] = Scalar(self._first_time.vals, remask)
        time[tstep_int.vals >= self._steps] = Scalar(self._last_time.vals,remask)
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

        # Reverse the order of the indices, but handle the top carefully
        tstep_int = tstep.int(self.shape[0], remask=remask, inclusive=inclusive,
                              shift=shift)
            # Note: If shift is True, the end of the last time step will map into
            # the first time step, as intended. If shift is False, the end of the
            # last time step will map into a negative time step instead.

        reversed_tstep = self._max_step - tstep_int

        return self._cadence.time_range_at_tstep(reversed_tstep, remask=False,
                                                 inclusive=False)
            # inclusive=False above because reversed_tstep == self._steps where
            # tstep_int == -1, which must be excluded. remask=False here because
            # the input has already been properly masked.

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

        tstep = self._cadence.tstep_at_time(time, remask=remask, derivs=derivs,
                                            inclusive=inclusive)

        # Only the order of whole steps is reversed; within a step, time still increases
        # with the index, so the fractional part carries over unchanged. This is the
        # inverse of time_at_tstep, which reverses the integer part the same way.
        tstep_int = tstep.int(self._steps, remask=False, inclusive=inclusive, shift=True)
            # Note: Because shift=True, the end of the underlying cadence maps into its
            # last time step with a fractional part of 1, which becomes the end of the
            # first time step here. remask=False because the input is already masked.

        return (self._max_step - tstep_int) + (tstep - tstep_int)

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

        (tstep_min,
         tstep_max) = self._cadence.tstep_range_at_time(time, remask=remask,
                                                        inclusive=inclusive)
        return (self.shape[0] - tstep_max, self.shape[0] - tstep_min)

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
            ReversedCadence: The time-shifted cadence.
        """

        return ReversedCadence(self._cadence.time_shift(secs), self._axis)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For ReversedCadence, this is accomplished by making the underlying cadence
        continuous.

        Returns:
            ReversedCadence: The continuous cadence.
        """

        return ReversedCadence(self._cadence.as_continuous(), self._axis)

##########################################################################################
