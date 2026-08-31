##########################################################################################
# oops/cadence/instant.py
##########################################################################################

import numpy as np

from polymath     import Scalar
from oops.cadence import Cadence


class Instant(Cadence):
    """TODO: This is a work in progress. Not fully tested. To be used by the InSitu
    Observation subclass. DO NOT USE.

    A Cadence subclass that represents the timing of an observation as a Scalar time of
    arbitrary shape.
    """

    def __init__(self, tdb):
        """Constructor for an Instant.

        Parameters:
            tdb (Scalar): A time Scalar in seconds TDB.

        Raises:
            ValueError: If every time in `tdb` is masked.
        """

        self._tdb = Scalar.as_scalar(tdb, recursive=False).as_float()

        # Work with the unmasked times in raveled index order
        vals = np.asarray(self._tdb.vals).ravel()
        if np.any(self._tdb.mask):
            vals = vals[np.asarray(self._tdb.antimask).ravel()]

        if vals.size == 0:
            raise ValueError('Instant tdb input must include at least one unmasked time')

        self.shape = self._tdb.shape
        self.time = (float(vals.min()), float(vals.max()))
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.time[1]

        # Each time step is instantaneous, so the cadence is never continuous, and a
        # time falls in more than one time step only where a time is tabulated twice.
        self.is_continuous = False
        self.is_unique = bool(np.unique(vals).size == vals.size)

        # Strides are the intervals between consecutive times in raveled index order.
        if vals.size > 1:
            tstrides = np.abs(np.diff(vals))
            self.min_tstride = float(tstrides.min())
            self.max_tstride = float(tstrides.max())
        else:
            self.min_tstride = 0.
            self.max_tstride = 0.

    def __getstate__(self):
        self.refresh()
        return (self._tdb,)

    def __setstate__(self, state):
        self.__init__(*state)
        self.freeze()

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values via interpolation.

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

        return self._tdb     #### Shouldn't this be self._tdb[tstep.int()]?

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        An Instant has zero duration, so the two returned times are equal.

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

        return (self._tdb, self._tdb)     ### Same comment as above

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

        return Scalar(np.zeros(self.shape), self._tdb != time)

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

        Raises:
            NotImplementedError: This method is not implemented for an Instant.
        """

        ### TBD
        raise NotImplementedError('not implemented')

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

        return Scalar.as_scalar(time, recursive=False) != self._tdb

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            Instant: The time-shifted cadence.
        """

        return Instant(self._tdb + secs)

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        Returns:
            Instant: The continuous cadence.
        """

        instant = Instant(self._tdb)
        instant.is_continuous = True
        return instant

##########################################################################################
