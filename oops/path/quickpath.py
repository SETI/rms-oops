##########################################################################################
# oops/path/quickpath_.py
##########################################################################################

import numpy as np
import scipy.interpolate as interp

from polymath        import Scalar, Vector3
from oops.config     import QUICK, LOGGING
from oops.event      import Event
from oops.path.path_ import Path
import oops.mutable as mutable


class QuickPath(Path):
    """QuickPath returns positions and velocities by interpolating another Path.
    """

    def __init__(self, path, tmin, tmax, quickdict):
        """Constructor for a QuickPath.

        Parameters:
            path (Path): The Path object that this QuickPath will emulate.
            tmin (float): The earliest time to tabulate in this QuickPath.
            tmax (float): The latest time to tabulate in this QuickPath.
            quickdict (dict): A dictionary containing all the QuickPath parameters.
        """

        path = Path.as_path(path)
        if path._shape != ():
            raise ValueError('shape of QuickPath must be ()')
        if isinstance(path, QuickPath):
            raise ValueError('QuickPath cannot be constructed from another QuickPath')

        mutable.refresh(path)
        self._slowpath = path
        self._waypoint = path._waypoint
        self._primary  = path._primary
        self._path_id  = path._path_id
        self._origin   = path._origin
        self._frame    = path._frame
        self._shape    = ()

        # Expand the time limits a little bit and round them to multiples of tstep
        tstep = quickdict['path_time_step']
        extend = quickdict['path_time_extension']
        extras = int(quickdict['path_extra_steps'])
        self._input_tmin = tmin
        self._input_tmax = tmax
        self._tstep = tstep
        self._tmin = tstep * ((tmin - extend) // tstep - extras)
        self._tmax = tstep * ((tmax + extend) // tstep + extras + 1)
        self._quickdict = quickdict

        mutable.refresh(self)

        # Test the precision
        precision = quickdict['path_self_check']
        if precision is not None:
            time = self._times[:-1] + self._tstep/2.        # halfway points
            true_event = self._slowpath.event_at_time(time, quick=False)
            (pos, vel) = self._interpolate_pos_vel(time)

            # Check largest fractional error
            dpos = (true_event.pos - pos).norm() / (true_event.pos).norm()
            dvel = (true_event.vel - vel).norm() / (true_event.vel).norm()
            error = max(np.max(dpos.vals), np.max(dvel.vals))
            if error > precision:
                raise ValueError(f'precision failure: {error:.3f} > {precision}')

    def _refresh(self):

        times = np.arange(self._tmin, self._tmax + self._tstep/2., self._tstep)
        self._steps = len(times)
        self._events = self._slowpath.event_at_time(times, quick=False)
        self._times = times
        self._spline_setup()

    def _spline_setup(self):
        """Set up the internal tabulation to be interpolated, based on `_times` and
        `_events`.
        """

        KIND = 3
        self._pos_x = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.pos.vals[:, 0], k=KIND)
        self._pos_y = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.pos.vals[:, 1], k=KIND)
        self._pos_z = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.pos.vals[:, 2], k=KIND)

        self._vel_x = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.vel.vals[:, 0], k=KIND)
        self._vel_y = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.vel.vals[:, 1], k=KIND)
        self._vel_z = interp.InterpolatedUnivariateSpline(
                                            self._times,
                                            self._events.vel.vals[:, 2], k=KIND)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        if self.pickle_quickpath_details:
            return self.__dict__
        else:
            return (self._slowpath, self._input_tmin, self._input_tmax, self._quickdict)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self.__init__(*state)
        else:
            self.__dict__ = state
        mutable.freeze(self)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, quick=False):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Event): The Event object containing (at least) the time, position, and
                velocity on the Path.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        (pos, vel) = self._interpolate_pos_vel(time)
        return Event(time, (pos, vel), self._origin, self._frame)

    def _interpolate_pos_vel(self, time, collapse_threshold=None):

        if collapse_threshold is None:
            collapse_threshold = \
                self._quickdict['quickpath_linear_interpolation_threshold']

        # `time` can only be a 1-D array in the splines
        time = Scalar.as_scalar(time)
        tflat = Scalar.as_scalar(time).flatten()
        if tflat.size == 0:
            vector3 = Vector3(np.ones(time.shape + (3,)), True).as_readonly()
            return (vector3, vector3)

        tflat_max = tflat.max(builtins=True)
        tflat_min = tflat.min(builtins=True)
        time_diff = tflat_max - tflat_min

        pos = np.empty(tflat.shape + (3,))
        vel = np.empty(tflat.shape + (3,))

        # If the time range is small, we only need to do linear interpolation.
        if time_diff <= collapse_threshold:

            # Create a time scalar just containing the end points
            tflat2 = Scalar([tflat_min, tflat_max])

            pos_x = self._pos_x(tflat2.vals)
            pos_y = self._pos_y(tflat2.vals)
            pos_z = self._pos_z(tflat2.vals)
            vel_x = self._vel_x(tflat2.vals)
            vel_y = self._vel_y(tflat2.vals)
            vel_z = self._vel_z(tflat2.vals)

            if time_diff == 0.:
                pos[..., 0] = pos_x[0]
                pos[..., 1] = pos_y[0]
                pos[..., 2] = pos_z[0]
                vel[..., 0] = vel_x[0]
                vel[..., 1] = vel_y[0]
                vel[..., 2] = vel_z[0]
            else:
                frac = (tflat.vals - tflat_min) / time_diff
                pos[..., 0] = pos_x[0] + frac * (pos_x[1] - pos_x[0])
                pos[..., 1] = pos_y[0] + frac * (pos_y[1] - pos_y[0])
                pos[..., 2] = pos_z[0] + frac * (pos_z[1] - pos_z[0])
                vel[..., 0] = vel_x[0] + frac * (vel_x[1] - vel_x[0])
                vel[..., 1] = vel_y[0] + frac * (vel_y[1] - vel_y[0])
                vel[..., 2] = vel_z[0] + frac * (vel_z[1] - vel_z[0])

        else:
            # Evaluate the positions and velocities using the splines
            pos[..., 0] = self._pos_x(tflat.vals)
            pos[..., 1] = self._pos_y(tflat.vals)
            pos[..., 2] = self._pos_z(tflat.vals)

            vel[..., 0] = self._vel_x(tflat.vals)
            vel[..., 1] = self._vel_y(tflat.vals)
            vel[..., 2] = self._vel_z(tflat.vals)

        # Return the positions and velocities
        return (Vector3(pos, tflat.mask).reshape(time.shape),
                Vector3(vel, tflat.mask).reshape(time.shape))

    ######################################################################################
    # QuickPath API
    ######################################################################################

    def extend(self, tmin, tmax):
        """Modify this QuickPath to accommodate a new, extended time interval.

        Parameters:
            tmin (float): The new earliest time to tabulate in this QuickFrame.
            tmax (float): The new latest time to tabulate in this QuickFrame.
        """

        # If the interval fits inside already, we're done
        if tmin >= self._tmin and tmax <= self._tmax:
            return

        # Extend the interval
        extend = self.quickdict('frame_time_extension')
        extras = int(self.quickdict('frame_extra_steps'))
        if tmin < self._tmin:
            self._input_tmin = tmin
            tmin = self._tstep * ((tmin - extend) // self._tstep - extras)
            time0 = np.arange(tmin, self._tmin, self._tstep)
            event0 = self._slowpath.event_at_time(time0)
            count0 = len(time0)
        else:
            tmin = self._tmin
            count0 = 0

        if tmax > self._tmax:
            self._input_tmax = tmax
            tmax = self._tstep * ((tmax + extend) // self._tstep + extras + 1)
            time1 = np.arange(self._tmax + self._tstep, tmax + self._tstep/2.,
                              self._tstep)
            event1 = self._slowpath.event_at_time(time1)
            count1 = len(time1)
        else:
            tmax = self._tmax
            count1 = 0

        if count0 + count1 == 0:
            return

        # Allocate the new arrays
        old_size = self._times.size
        new_size = old_size + count0 + count1

        pos_vals = np.empty((new_size, 3))
        vel_vals = np.empty((new_size, 3))

        # Copy the new arrays
        if count0 > 0:
            pos_vals[0:count0] = event0.pos.vals
            vel_vals[0:count0] = event0.vel.vals

        if count1 > 0:
            pos_vals[-count1:] = event1.pos.vals
            vel_vals[-count1:] = event1.vel.vals
        else:
            count1 = -new_size      # this makes the indexing below work correctly

        pos_vals[count0:-count1] = self.events.pos.vals
        vel_vals[count0:-count1] = self.events.vel.vals

        # Generate the new events
        self._times = np.arange(tmin, tmax + self._tstep/2., self._tstep)
        self._tmin = tmin
        self._tmax = tmax
        self._events = Event(Scalar(self._times), (Vector3(pos_vals), Vector3(vel_vals)),
                             self._events._origin, self._events._frame)

        # Regenerate the splines
        self._spline_setup()

    @staticmethod
    def for_path(path, time, *, quick=None):
        """A new QuickPath that approximates this path within given time limits.

        A QuickPath operates by sampling the given path and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed
        up performance when the same path must be evaluated many times, e.g.,
        for every pixel of an image.

        Parameters:
            path (Path): The Path to be approximated.
            time (Scalar or tuple): The set of times at which the frame is to be
                evaluated. This can simply be a tuple (`tmin`, `tmax`) defining the
                beginning and end times.
            quick (dict or bool, optional): If False, no QuickPath is created and self is
                returned; if a dictionary, then the values provided override the values in
                the default dictionary QUICK.dictionary, and the merged dictionary is
                used.

        Notes:
            QuickPaths generated by this function are saved as a list inside
            `path._quickpaths`. If a pre-existing QuickPath that covers the time range is
            found in this list, it is returned rather than constructing a new QuickPath.
            If a QuickPath is found in the list that partially covers the time range, that
            QuickPath is extended to cover the full range and returned.
        """

        if not path._USE_QUICKPATHS:
            return path

        if path._shape:     # the Path must be shapeless
            return path

        # Make sure a QuickPath has been requested
        if quick is None:
            quick = {}
        if not isinstance(quick, dict):
            return path

        # Obtain the local QuickFrame dictionary
        quickdict = QUICK.dictionary
        if quick:
            quickdict = quickdict.copy()
            quickdict.update(quick)

        if not quickdict['use_quickpaths']:
            return path

        # Determine the time interval
        time = Scalar.as_scalar(time)
        tmin = time.min(builtins=True)
        tmax = time.max(builtins=True)
        if tmin == Scalar.MASKED:
            return path

        # Initialize the cache if it is missing
        if not hasattr(path, '_quickpaths'):
            path._quickpaths = []

        # If an existing QuickPath covers the whole time range, just return it
        for quickpath in path._quickpaths:
            if tmin >= quickpath._tmin and tmax <= quickpath._tmax:
                if LOGGING.quickpath_creation:
                    LOGGING.diagnostic(f'Re-using QuickPath for {path}: '
                                       f'{tmin:.3f}, {tmax:.3f})')
                return quickpath

        # This is a quick-and-dirty algorithm to determine whether the use of a QuickPath
        # is worth the effort relative to using the given Path.
        #
        # We assume that constructing the QuickPath carries with it a level of overhead
        # equivalent to _OVERHEAD evaluations.
        #
        # Once constructed, the QuickPath will be evaluated at least _MIN_EVALUATIONS
        # times, and each evaluation will be _SPEEDUP times faster.
        #
        # If the improvement is less than a factor of _MIN_SAVINGS, we might as well use
        # the original path.
        #
        # These are WAGs but make sure that, under reasonable circumstances, a QuickPath
        # is created:
        _OVERHEAD = 200
        _MIN_EVALUATIONS = 1000
        _SPEEDUP = 10.
        _MIN_SAVINGS = 0.2

        # Estimate the number of Frame evaluations needed by the QuickFrame
        tstep = quickdict['path_time_step']
        extend = quickdict['path_time_extension']
        extras = int(quickdict['path_extra_steps'])

        evaluations = max(time.size, _MIN_EVALUATIONS)
        savings_per_evaluation = 1. - 1./_SPEEDUP

        # See if any QuickPath can be efficiently extended
        for quickpath in path._quickpaths:

            # If there's no overlap, skip it
            if (quickpath._tmin > tmax + tstep) or (quickpath._tmax < tmin - tstep):
                continue

            # Otherwise, check the effort involved
            new_duration = (max(tmax, quickpath._tmax) - min(tmin, quickpath._tmin)
                            + 2*extend)
            new_steps = (new_duration/tstep + 2*extras) - quickpath._steps
            overhead = _OVERHEAD + new_steps
            if savings_per_evaluation - overhead/evaluations >= _MIN_SAVINGS:
                if LOGGING.quickpath_creation:
                    LOGGING.diagnostic(f'Extending QuickPath for {path}: '
                                       f'{tmin:.3f}, {tmax:.3f})')
                quickpath.extend((tmin, tmax))
                return quickpath

        # Otherwise, construct a new QuickFrame
        steps = (tmax - tmin + 2*extend)/tstep + 2*extras
        overhead = _OVERHEAD + steps
        if savings_per_evaluation - overhead/evaluations >= _MIN_SAVINGS:
            if LOGGING.quickpath_creation:
                LOGGING.diagnostic(f'New QuickPath for {path}: {tmin:.3f}, {tmax:.3f})')

            result = QuickPath(path, tmin, tmax, quickdict)
            if len(path._quickpaths) >= quickdict['quickpath_cache_size']:
                path._quickpaths = [result] + path._quickpaths[:-1]
            else:
                path._quickpaths = [result] + path._quickpaths
            return result

        return path

##########################################################################################

Path._PATH_SUBCLASSES.append(QuickPath)
Path._QuickPath = QuickPath

##########################################################################################
