##########################################################################################
# oops/cadence/instant.py
##########################################################################################

import numpy as np

from polymath     import Boolean, Qube, Scalar, Vector
from oops.cadence import Cadence


class Instant(Cadence):
    """A Cadence subclass that represents the timing of an observation as a Scalar time of
    arbitrary shape.

    Every time step is a single moment rather than an interval, so this cadence has gaps
    between its time steps and samples no time in between. Time steps are indexed by the
    shape of the time Scalar: a Scalar index for a 1-D cadence, and a Pair or Vector with
    one component per axis otherwise. A cadence of shape () has one time step.
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

    ######################################################################################
    # Support for converting between time steps and times
    ######################################################################################

    def _index_at_tstep(self, tstep, *, remask=False, inclusive=True):
        """The array index into the times for the given time step.

        Indices beyond the limits of the cadence are clipped to the nearest edge.

        Parameters:
            tstep (Scalar or Pair): Time step index values, with one component per axis
                of this cadence's shape.
            remask (bool, optional): True to mask values outside the cadence.
            inclusive (bool, optional): True to treat the largest index as part of this
                Cadence; False to exclude it.

        Returns:
            Scalar or Vector: The integer index, always within the shape of the times.
        """

        if len(self.shape) == 1:
            return Scalar.as_scalar(tstep, recursive=False).int(
                            top=self.shape[0], remask=remask, clip=True,
                            inclusive=inclusive)

        return Vector.as_vector(tstep, recursive=False).int(
                            self.shape, remask=remask, clip=True, inclusive=inclusive)

    def _match_at_time(self, time):
        """The index of the first time step sampling each of the given times.

        A time step matches only where it is unmasked and its time is exactly equal to
        the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.

        Returns:
            tuple[ndarray, ndarray]: The raveled index of the first matching time step,
            zero where a time is not sampled; and a boolean array, True where a time is
            sampled.
        """

        table = np.asarray(self._tdb.vals, dtype='float64').ravel()
        antimask = np.asarray(self._tdb.antimask).ravel()

        vals = np.asarray(time.vals, dtype='float64')
        matches = (vals[..., np.newaxis] == table) & antimask

        return (matches.argmax(axis=-1), matches.any(axis=-1))

    ######################################################################################
    # Standard Cadence methods
    ######################################################################################

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        Each time step of an Instant is instantaneous, so a non-integer time step is
        truncated to the step that contains it rather than interpolated. A time step
        beyond the limits of the cadence returns the time at the nearest edge.

        Parameters:
            tstep (Scalar or Pair): Time step index values, with one component per axis
                of this cadence's shape. It is ignored if this cadence has shape (),
                because then it has only one time step.
            remask (bool, optional): True to mask values outside the time limits.
            derivs (bool, optional): Ignored. The returned time carries no derivatives,
                because the time does not vary within a time step.
            inclusive (bool, optional): True to treat the largest time step index as part
                of this Cadence; False to exclude it.

        Returns:
            Scalar: Time in seconds TDB, masked where the time step is masked and where
            the time step it selects is masked.
        """

        if not self.shape:
            return self._tdb

        return self._tdb[self._index_at_tstep(tstep, remask=remask,
                                              inclusive=inclusive)]

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        An Instant has zero duration, so the two returned times are equal.

        Parameters:
            tstep (Scalar or Pair): Time step index values, with one component per axis
                of this cadence's shape.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the largest time step index as part
                of this Cadence; False to exclude it.
            shift (bool, optional): Ignored. Time step indices are clipped to the shape
                of the cadence, so the largest index needs no shift.

        Returns:
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the
            index values, in seconds TDB. The two are equal.
        """

        time = self.time_at_tstep(tstep, remask=remask, inclusive=inclusive)

        return (time, time)

    def tstep_at_time(self, time, *, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        An Instant samples isolated moments, so only a time exactly equal to one of them
        has a time step. Any other time is masked, whether or not `remask` is True,
        because no time step describes it. Where the same time appears more than once,
        the first of the matching time steps is returned.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): Ignored. A time that this cadence does not sample is
                always masked.
            derivs (bool, optional): Ignored. The returned time step carries no
                derivatives, because it does not vary with time.
            inclusive (bool, optional): Ignored. Every time step of this cadence is a
                single moment, which is always treated as part of the cadence.

        Returns:
            Scalar or Pair: Time step index values, with one component per axis of this
            cadence's shape, masked where the time is not sampled. A cadence of shape ()
            has one time step, whose index is zero.
        """

        time = Scalar.as_scalar(time, recursive=False)
        (raveled, found) = self._match_at_time(time)
        mask = Qube.or_(time.mask, np.logical_not(found))

        if len(self.shape) > 1:
            indices = np.stack(np.unravel_index(raveled, self.shape), axis=-1)
            tstep = Vector(indices, mask)
            return tstep.to_pair() if len(self.shape) == 2 else tstep

        if not self.shape:
            raveled = np.zeros(time.shape, dtype='int64')

        return Scalar(raveled, mask)

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within this
                Cadence.
            inclusive (bool, optional): Ignored. Every time step of this cadence is a
                single moment, which is always treated as part of the cadence.

        Returns:
            tuple[Scalar or Pair, Scalar or Pair]: The range of time step indices active
            at the given `time`, as (first, last+1); the upper limit is excluded. Values
            are always within the allowed range for the cadence, regardless of any mask.
            A time that this cadence does not sample yields an empty range, meaning that
            the second value equals the first.
        """

        time = Scalar.as_scalar(time, recursive=False)
        (raveled, found) = self._match_at_time(time)
        mask = Qube.or_(time.mask, np.logical_not(found)) if remask else time.mask
        span = found.astype('int64')

        if len(self.shape) > 1:
            first = np.stack(np.unravel_index(raveled, self.shape), axis=-1)
            tsteps = (Vector(first, mask), Vector(first + span[..., np.newaxis], mask))
            if len(self.shape) == 2:
                return (tsteps[0].to_pair(), tsteps[1].to_pair())
            return tsteps

        if not self.shape:
            raveled = np.zeros(time.shape, dtype='int64')

        return (Scalar(raveled, mask), Scalar(raveled + span, mask))

    def time_is_outside(self, time, *, inclusive=True):
        """A Boolean mask of times that fall outside the cadence.

        An Instant samples isolated moments, so every time other than one of those
        moments falls outside it.

        Parameters:
            time (Scalar): Times in seconds TDB.
            inclusive (bool, optional): Ignored. Every time step of this cadence is a
                single moment, which is always treated as inside.

        Returns:
            Boolean: True where `time` is not sampled by the cadence, masked where `time`
            is masked.
        """

        time = Scalar.as_scalar(time, recursive=False)
        (_, found) = self._match_at_time(time)

        return Boolean(np.logical_not(found), time.mask)

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
