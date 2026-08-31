##########################################################################################
# oops/cadence/cadence_
##########################################################################################

from polymath import Scalar, Pair
from oops.mutable import Mutable


class Cadence(Mutable):
    """Cadence is an abstract class that defines the timing of an observation.

    Properties:
        * time (tuple): The start time and end time of the observation overall, in
          seconds TDB.
        * midtime (float): The mid-time of the observation, in seconds TDB.
        * lasttime (float): The start time of the last time step, in seconds TDB.
        * shape (tuple): Integers defining the shape of the indices.
        * is_continuous (bool): True if the cadence contains no gaps in time between the
          start and end.
        * is_unique (bool): True if no times inside the cadence are associated with more
          than one time step.
        * min_tstride (float): Minimum absolute value of the time interval between one
          tstep and the next.
        * max_tstride (float): Maximum absolute value of the time interval between one
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
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Scalar): Times in seconds TDB.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_at_tstep is not '
                                  'implemented')

    def time_range_at_tstep(self, tstep, *, remask=False, inclusive=True, shift=True):
        """The range of times for the given time step.

        In multidimensional cadences, indexing beyond the dimensions of the cadence
        returns the time range at the nearest edge.

        Parameters:
            tstep (Pair): Time step index values.
            remask (bool, optional): True to mask values outside the time limits.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.
            shift (bool, optional): True to shift the end of the last time step (with
                index==shape) into the previous time step.

        Returns:
            (tuple): (time_min, time_max), where:

            * `time_min` (Scalar): Defining the minimum time associated with the index. It
              is given in seconds TDB.
            * `time_max` (Scalar): Defining the maximum time value.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_range_at_tstep is not '
                                  'implemented')

    def tstep_at_time(self, time, *, remask=False, derivs=False, inclusive=True):
        """Time step for the given time.

        This method returns non-integer time steps via interpolation.

        In multidimensional cadences, times before first time step refer to the first;
        times after the last time step refer to the last.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within the
                cadence.
            derivs (bool, optional): True to include derivatives of time in the returned
                tstep.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (Scalar or Pair): Time step indices.
        """

        raise NotImplementedError(f'{type(self).__name__}.tstep_at_time is not '
                                  'implemented')

    def tstep_range_at_time(self, time, *, remask=False, inclusive=True):
        """Integer range of time steps active at the given time.

        Parameters:
            time (Scalar): Times in seconds TDB.
            remask (bool, optional): True to mask time values not sampled within the
                cadence.
            inclusive (bool, optional): True to treat the end time of the cadence as part
                of the cadence; False to exclude it.

        Returns:
            (tuple): (tstep_min, tstep_max) tstep_min   minimum Scalar or Pair time step
                containing the given time. tstep_max   maximum Scalar or Pair time step
                containing the given time (inclusive). All returned indices will be in the
                allowed range for the cadence, inclusive, regardless of mask. If the time
                is not inside the cadence, tstep_max < tstep_min.
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
            (bool): Array indicating which time values are not sampled by the cadence.
        """

        time = Scalar.as_scalar(time, recursive=False)

        # Default behavior is to treat all times between start and stop as
        # inside. Discontinuous subclasses need to override.

        if inclusive:
            return time.tvl_lt(self.time[0]) | time.tvl_gt(self.time[1])
        else:
            return time.tvl_lt(self.time[0]) | time.tvl_ge(self.time[1])

    def time_shift(self, secs):
        """Construct a duplicate of this Cadence with all times shifted by given amount.

        Parameters:
            secs (float): Seconds to shift the time later.
        """

        raise NotImplementedError(f'{type(self).__name__}.time_shift is not implemented')

    def as_continuous(self):
        """Construct a shallow copy of this Cadence, forced to be continuous."""

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
            (bool): Array indicating which time values are sampled by the cadence.
        """

        return self.time_is_outside(time, inclusive=inclusive).logical_not()

    def tstride_at_tstep(self, tstep, sign=1, *, remask=False):
        """The time interval(s) between the times of adjacent time steps.

        Parameters:
            tstep (Scalar or Pair): Time step index, which need not be integral.
            sign (int, optional): +1 for the time interval to the next time step; -1 for
                the time interval since the previous time step.
            remask (bool, optional): True to mask time tsteps that are out of range.

        Returns:
            (Scalar or Pair): Strides in seconds.
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
            tstride = tstride.remask(tstep.mask)
            return tstride

        raise NotImplementedError(f'{type(self).__name__}.tstride_at_tstep is not '
                                  f'implemented for {len(self.shape)}-D cadences')

##########################################################################################
