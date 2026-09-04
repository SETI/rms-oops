##########################################################################################
# oops/cadence/timeshift.py
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
            self._link = arg
        else:
            self._dt = arg
            self._link = None

        self._cadence = cadence
        self._refresh()

        self.shape = cadence.shape
        self.is_continuous = cadence.is_continuous
        self.is_unique = cadence.is_unique
        self.min_tstride = cadence.min_tstride
        self.max_tstride = cadence.max_tstride

    @property
    def dt(self):
        """The time shift in seconds. A positive value shifts times later."""
        return self._dt

    @property
    def link(self):
        """The object to which this one is linked, or None if it is unlinked."""
        return self._link

    def _source(self):
        """The original source of the time shift, or self if there is none.
        """
        return self._link._source() if self._link else self

    ######################################################################################
    # Fittable support
    ######################################################################################

    nparams = 1
    is_initialize = True

    @property
    def params(self):
        """The fitted parameters, the time shift in seconds as a tuple of one float."""

        return (self._dt,)

    def _set_params(self, params):
        """Update the time shift in seconds.

        If this object is linked to another, the time offset of the linked object is also
        redefined.
        """

        if self._link:
            self._link.set_params(params)
            self._dt = self._link.dt
        else:
            self._dt = params[0]

    def _refresh(self):
        """Update the internals."""

        if self._link:
            self._link._refresh()
            self._dt = self._link.dt

        self.time = (self._cadence.time[0] + self._dt, self._cadence.time[1] + self._dt)
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self._cadence.lasttime + self._dt

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._dt, self._cadence)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    ######################################################################################
    # Cadence API
    ######################################################################################

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time at the nearest edge of the cadence's shape.

        Parameters:
            tstep (Scalar or Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): True to include derivatives of tstep in the returned
                time.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar: Time in seconds TDB.
        """

        return (self._cadence.time_at_tstep(tstep=tstep, remask=remask, derivs=derivs,
                                           inclusive=inclusive)
                + self._dt)

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time range at the nearest edge.

        Parameters:
            tstep (Scalar or Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index == shape) into the previous time step.

        Returns:
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the index
            values, in seconds TDB.
        """

        times = self._cadence.time_range_at_tstep(tstep, remask=remask,
                                                 inclusive=inclusive, shift=shift)
        return (times[0] + self._dt, times[1] + self._dt)

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
            Scalar or Pair: Time step index values.
        """

        return self._cadence.tstep_at_time(time - self._dt, remask=remask, derivs=derivs,
                                          inclusive=inclusive)

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            inclusive (bool, optional): True to treat the end time as part of this
                Cadence; False to exclude it.

        Returns:
            tuple[Scalar or Pair, Scalar or Pair]: The range of time step indices active
            at the given `time`, as (first, last+1); the upper limit is excluded. Values
            are always within the allowed range for the cadence, regardless of any mask.
            If `time` is not sampled by the cadence, the range is empty, meaning that the
            second value equals the first.
        """

        time = Scalar.as_scalar(time)
        return self._cadence.tstep_range_at_time(time - self._dt, remask=remask,
                                                inclusive=inclusive)

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

        time = Scalar.as_scalar(time)
        return self._cadence.time_is_outside(time - self._dt, inclusive=inclusive)

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            TimeShift: The time-shifted cadence.
        """

        return TimeShift(self._link or self._dt, self._cadence.time_shift(secs))

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        For TimeShift, this is accomplished by making the underlying cadence continuous.

        Returns:
            TimeShift: The continuous cadence.
        """

        return TimeShift(self._link or self._dt, self._cadence.as_continuous())

##########################################################################################
