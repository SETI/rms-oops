##########################################################################################
# oops/cadence/timeshift.py: Class TimeShift
##########################################################################################

from polymath      import Scalar
from oops.cadence  import Cadence
from oops.fittable import Fittable


class TimeShift(Cadence, Fittable):
    """A Fittable time shift applied to another Cadence object."""

    def __init__(self, arg, /, cadence):
        """Constructor for a TimeShift.

        Parameters:
            arg (float, TimeShift, FrameShift, or PathShift): The initial time shift in
                seconds. A positive value shifts times later. Alternatively, if another
                time-shifted object is given, this object's time shift will always match
                that of the argument.
            cadence (Cadence): The Cadence object to be shifted.
        """

        if hasattr(arg, 'dt'):
            self.link = arg
        else:
            self.dt = arg
            self.link = None

        self.cadence = cadence
        self._refresh()

        self.shape = cadence.shape
        self.is_continuous = cadence.is_continuous
        self.is_unique = cadence.is_unique
        self.min_tstride = cadence.min_tstride
        self.max_tstride = cadence.max_tstride

    def _source(self):
        """The original source of the time shift if this object is linked to another;
        otherwise, self.
        """
        return self.link._source() if self.link else self

    ######################################################################################
    # Fittable support
    ######################################################################################

    nparams = 1
    is_initialize = True

    @property
    def params(self):
        return (self.dt,)

    def _set_params(self, params):
        """Update the time shift in seconds.

        If this object is linked to another, the time offset of the linked object is also
        redefined.
        """

        if self.link:
            self.link.set_params(params)
            self.dt = self.link.dt
        else:
            self.dt = params[0]

    def _refresh(self):
        """Update the internals."""

        if self.link:
            self.link._refresh()
            self.dt = self.link.dt

        self.time = (self.cadence.time[0] + self.dt, self.cadence.time[1] + self.dt)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.cadence.lasttime + self.dt

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self.dt, self.cadence)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    ######################################################################################
    # Cadence API
    ######################################################################################

    def time_at_tstep(self, tstep, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time at the nearest edge of the cadence's shape.

        Parameters:
            tstep (Scalar, Pair, array-like, float, or int): Time step index, 1-D or 2-D.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Scalar): Time(s) in seconds TDB.
        """

        return (self.cadence.time_at_tstep(tstep=tstep, remask=remask, derivs=derivs,
                                           inclusive=inclusive)
                + self.dt)

    def time_range_at_tstep(self, tstep, remask=False, inclusive=True,
                            shift=True):
        """The range of times for the given time step.

        In multidimensional cadences, indexing beyond the dimensions of the
        cadence returns the time range at the nearest edge.

        Parameters:
            tstep (Scalar, Pair, array-like, float, or int): Time step index, 1-D or 2-D.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index==shape) into the previous time step.

        Returns:
            (tuple): Two Scalars defining the minimum and maximum times associated with
            the index. It is given in seconds TDB.
        """

        times = self.cadence.time_range_at_tstep(tstep, remask=remask,
                                                 inclusive=inclusive, shift=shift)
        return (times[0] + self.dt, times[1] + self.dt)

    def tstep_at_time(self, time, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps via interpolation.

        In multidimensional cadences, times before first time step refer to the
        first; times after the last time step refer to the last.

        Parameters:
            time (Scalar, array-like, float, or int): Time(s) in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Scalar or Pair): Time step index or indices.
        """

        return self.cadence.tstep_at_time(time - self.dt, remask=remask, derivs=derivs,
                                          inclusive=inclusive)

    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar, array-like, float, or int): Time(s) in seconds TDB.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (tuple): Two Scalars defining the minimum and maximum tstep values.

        Notes:
            All returned indices will be in the allowed range for the cadence, inclusive,
            regardless of mask. If the time is not inside the cadence, `tstep_max <
            tstep_min`.
        """

        time = Scalar.as_scalar(time)
        return self.cadence.tstep_range_at_time(time - self.dt, remask=remask,
                                                inclusive=inclusive)

    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Parameters:
            time (Scalar, array-like, float, or int): Time(s) in seconds TDB.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Boolean): True where time values are not sampled by the Cadence.
        """

        time = Scalar.as_scalar(time)
        return self.cadence.time_is_outside(time - self.dt, inclusive=inclusive)

    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given amount.

        Parameters:
            secs (float): The number of seconds to shift the time later.
        """

        return TimeShift(self.link or self.dt, self.cadence.time_shift(secs))

    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous."""

        return TimeShift(self.link or self.dt, self.cadence.as_continuous())

##########################################################################################
