##########################################################################################
# oops/cadence/cadence_.py
##########################################################################################

from polymath import Scalar, Pair
from oops.mutable import Mutable


class Cadence(Mutable):
    """An abstract class defining the timing of an observation.

    Attributes:
        time (tuple): The start time and end time of the observation overall, in seconds
            TDB.
        midtime (float): The mid-time of the observation, in seconds TDB.
        lasttime (float): The start time of the last time step, in seconds TDB.
        shape (tuple): The shape of the array of time step indices.
        is_continuous (bool): True if the cadence contains no gaps in time between the
            start and end.
        is_unique (bool): True if no times inside the cadence are associated with more
            than one time step.
        min_tstride (float): Minimum absolute value of the time interval between one
            tstep and the next.
        max_tstride (float): Maximum absolute value of the time interval between one
            tstep and the next.
    """

    ######################################################################################
    # Methods to be defined for each Cadence subclass
    ######################################################################################

    def time_at_tstep(self, tstep, *, remask=False, derivs=False, inclusive=True):
        """The time associated with the given time step.

        This method supports non-integer time step values.

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

        raise NotImplementedError(f'{type(self).__name__}.time_at_tstep is not '
                                  'implemented')

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
            tuple[Scalar, Scalar]: The minimum and maximum times associated with the
            index values, in seconds TDB.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_range_at_tstep is not '
                                  'implemented')

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

        raise NotImplementedError(f'{type(self).__name__}.tstep_at_time is not '
                                  'implemented')

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

        raise NotImplementedError(f'{type(self).__name__}.tstep_range_at_time is not '
                                  'implemented')

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

        time = Scalar.as_scalar(time, recursive=False)

        # Default behavior is to treat all times between start and stop as
        # inside. Discontinuous subclasses need to override.

        if inclusive:
            return time.tvl_lt(self.time[0]) | time.tvl_gt(self.time[1])
        else:
            return time.tvl_lt(self.time[0]) | time.tvl_ge(self.time[1])

    def time_shift(self, secs):
        """A duplicate of this Cadence with all times shifted by the given amount.

        Parameters:
            secs (float): Seconds to shift the time later.

        Returns:
            Cadence: The time-shifted cadence.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_shift is not implemented')

    def as_continuous(self):
        """A shallow copy of this Cadence, forced to be continuous.

        Returns:
            Cadence: The continuous cadence.
        """

        raise NotImplementedError(f'{type(self).__name__}.as_continuous is not '
                                  'implemented')

    ######################################################################################
    # Methods probably not requiring overrides
    ######################################################################################

    def time_is_inside(self, time, *, inclusive=True):
        """A Boolean mask of times that fall inside the cadence.

        Parameters:
            time (Scalar): Times in seconds TDB.
            inclusive (bool, optional): True to treat the end time of an interval as
                inside; False to treat it as outside. The start time of an interval is
                always treated as inside.

        Returns:
            Boolean: True where `time` is sampled by the cadence.
        """

        return self.time_is_outside(time, inclusive=inclusive).logical_not()

    def tstride_at_tstep(self, tstep, sign=1, *, remask=False):
        """The time interval(s) between the times of adjacent time steps.

        Parameters:
            tstep (Scalar or Pair): Time step index, which need not be integral.
            sign (int, optional): +1 for the time interval to the next time step; -1 for
                the time interval since the previous time step.
            remask (bool, optional): True to mask tsteps that are out of range.

        Returns:
            Scalar or Pair: Strides in seconds.

        Raises:
            NotImplementedError: If the cadence has more than two dimensions.
        """

        if remask:
            time = self.time_at_tstep(tstep, remask=True)
            new_mask = time.mask
        else:
            new_mask = False

        if len(self.shape) == 1:
            tstep = Scalar.as_scalar(tstep, recursive=False)

            if sign < 0:
                tstep -= 1

            tstep = tstep.clip(0, self.shape[0]-1, remask=False)

            time0 = self.time_at_tstep(tstep  , remask=False)
            time1 = self.time_at_tstep(tstep+1, remask=False)

            tstride = time1 - time0
            tstride = tstride.remask_or(new_mask)
            return tstride

        if len(self.shape) == 2:
            tstep = Pair.as_pair(tstep, recursive=False).copy()
            (u,v) = tstep.to_scalars()                      # shared memory

            if sign < 0:
                u -= 1
                v -= 1

            u[u < 0] = 0
            v[v < 0] = 0

            utop = self.shape[0] - 1
            vtop = self.shape[1] - 1
            u[u > utop] = utop
            v[v > vtop] = vtop

            time0  = self.time_at_tstep(tstep, remask=False)
            time1u = self.time_at_tstep(tstep+(1,0), remask=False)
            time1v = self.time_at_tstep(tstep+(0,1), remask=False)
            tstride = Pair.from_scalars(time1u - time0, time1v - time0)
            tstride = tstride.remask_or(new_mask)
            return tstride

        raise NotImplementedError(f'{type(self).__name__}.tstride_at_tstep is not '
                                  f'implemented for {len(self.shape)}-D cadences')

##########################################################################################
