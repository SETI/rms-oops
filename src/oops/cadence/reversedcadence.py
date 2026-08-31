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
            axis (optional): Axis to reverse.
        """

        self.cadence = cadence
        if len(self.cadence.shape) != 1:
            raise ValueError('ReversedCadence must be based on a 1-D cadence')

        # Required attributes
        self.shape         = self.cadence.shape
        self.lasttime      = self.cadence.lasttime
        self.time          = self.cadence.time
        self.midtime       = self.cadence.midtime
        self.shape         = self.cadence.shape
        self.is_continuous = self.cadence.is_continuous
        self.is_unique     = self.cadence.is_unique
        self.min_tstride   = self.cadence.min_tstride
        self.max_tstride   = self.cadence.max_tstride

        # Used internally
        self.steps = self.cadence.shape[0]
        self._max_step = self.steps - 1

        # Beginning of new first time step; end of new last time step
        self._first_time = self.cadence.time_range_at_tstep(self._max_step)[0]
        self._last_time  = self.cadence.time_range_at_tstep(0)[1]

    def _refresh(self):
        """Update internals if self.cadence is Fittable."""
        self.time = self.cadence.time
        self.midtime = self.cadence.midtime
        self.lasttime = self.cadence.lasttime
        self._first_time = self.cadence.time_range_at_tstep(self._max_step)[0]
        self._last_time  = self.cadence.time_range_at_tstep(0)[1]

    def __getstate__(self):
        self.refresh()
        return (self.cadence,)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        Parameters:
            tstep (Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Scalar): Times in seconds TDB.
        """

        tstep = Scalar.as_scalar(tstep, recursive=derivs)

        # Reverse the order of the indices, but allow the fractional part to
        # increase within each time step.

        tstep_int = tstep.int(self.steps, remask=remask, inclusive=inclusive,
                                          shift=True)
            # Note: Because shift=True, the end of the last time step will map
            # into the first time step, yielding tstep_frac = 1 below,
            # regardless of whether it is to be included.

        reversed_tstep = self._max_step - tstep_int
        (time0,
         time1) = self.cadence.time_range_at_tstep(reversed_tstep, remask=False,
                                                   inclusive=False)
            # inclusive=False above because reversed_tstep == self.steps where
            # tstep_int == -1, which must be excluded. remask=False because the
            # input is already properly masked.

        tstep_frac = tstep - tstep_int
        time = time0 + tstep_frac * (time1 - time0)

        # Force out-of range tsteps to the start or end time
        time[tstep_int.vals < 0] = Scalar(self._first_time.vals, remask)
        time[tstep_int.vals >= self.steps] = Scalar(self._last_time.vals,remask)
        return time

    def time_range_at_tstep(self, tstep, remask=False, inclusive=True):
        """The range of times for the given time step.

        Parameters:
            tstep (Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (tuple): (time_min, time_max), where:

            * `time_min` (Scalar): Defining the minimum time associated with the index. It
              is given in seconds TDB.
            * `time_max` (Scalar): Defining the maximum time value.
        """

        tstep = Scalar.as_scalar(tstep, recursive=False)

        # Reverse the order of the indices, but handle the top carefully
        tstep_int = tstep.int(self.shape[0], remask=remask, inclusive=inclusive)
            # Note: If inclusive is True, the end of the last time step will map
            # into the first time step, as intended. If inclusive is False, the
            # end of the last time step will map into a negative time step, also
            # as intended.

        reversed_tstep = self._max_step - tstep_int

        return self.cadence.time_range_at_tstep(reversed_tstep,
                                                remask=False, inclusive=False)
            # inclusive=False above because reversed_tstep == self.steps where
            # tstep_int == -1, which must be excluded. remask=False here because
            # the input has already been properly masked.

    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within the
                cadence.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Pair): Time step index values.
        """

        tstep = self.cadence.tstep_at_time(time, remask=remask, derivs=derivs,
                                                 inclusive=inclusive)
        return self.shape[0] - tstep

    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within the
                cadence.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (tuple): (tstep_min, tstep_max) tstep_min   minimum Pair time step containing
                the given time. tstep_max   maximum Pair time step containing the given
                time (inclusive). All returned indices will be in the allowed range for
                the cadence, inclusive, regardless of mask. If the time is not inside the
                cadence, tstep_max < tstep_min.
        """

        (tstep_min,
         tstep_max) = self.cadence.tstep_range_at_time(time, remask=remask,
                                                       inclusive=inclusive)
        return (self.shape[0] - tstep_max, self.shape[0] - tstep_min)

    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Parameters:
            time (Scalar): Times in seconds TDB.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (bool): Array indicating which time values are not sampled by the cadence.
        """

        return self.cadence.time_is_outside(time, inclusive=inclusive)

    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given amount.

        Parameters:
            secs (float): Seconds to shift the time later.
        """

        return ReversedCadence(self.cadence.time_shift(secs))

    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous.

        For DualCadence, this is accomplished by forcing the stride of
        the short cadence to be continuous.
        """

        return ReversedCadence(self.cadence.as_continuous())

##########################################################################################
