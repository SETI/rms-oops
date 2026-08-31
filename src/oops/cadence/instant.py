##########################################################################################
# oops/cadence/instant.py
##########################################################################################

import numpy as np

from polymath     import Scalar
from oops.cadence import Cadence


class Instant(Cadence):
    """TODO: This is a work in progress. Not fully tested. To be used by the InSitu
    Observation subclass. DO NOT USE.

    A Cadence subclas that represents the timing of an observation as a Scalar time of
    arbitrary shape.
    """

    def __init__(self, tdb):
        """Constructor for an Instant.

        Parameters:
            tdb (Scalar): A time Scalar in seconds TDB.
        """

        self.tdb = Scalar.as_scalar(tdb, recursive=False).as_float()

        self.shape = self.tdb.shape
        self.time = (self.tdb.min(), self.tdb.max())
        self.midtime = 0.5 * (self.time[0] + self.time[1])
        self.lasttime = self.tdb.max()
        self.is_continuous = False

    def __getstate__(self):
        self.refresh()
        return (self.tdb)

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

        return self.tdb     #### Shouldn't this be self.tdb[tstep.int()]?

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

        return (self.tdb, self.tdb)     ### Same comment as above

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

        return Scalar(np.zeros(self.shape), self.tdb != time)

    def tstep_range_at_time(self, time, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within the
                cadence.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (tuple): (tstep_min, tstep_max) tstep_min   minimum integer time step of
                active range. tstep_max   maximum integer time step of active range. All
                returned values will be in the range (0, steps-1) inclusive. regardless of
                mask. If the time is not inside the cadence, tstep_max < tstep_min.
        """

        ### TDB
        raise NotImplementedError('not implemented')

    def time_is_outside(self, time, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        Parameters:
            time (Scalar): Times in seconds TDB.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (bool): Array indicating which time values are not sampled by the cadence.
        """

        return Scalar.as_scalar(time, recursive=False) != self.tdb

    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given amount.

        Parameters:
            secs (float): Seconds to shift the time later.
        """

        return Instant(self.tdb + secs)

    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous."""

        instant = Instant(self.tdb)
        instant.is_continuoue = True
        return instant

##########################################################################################
