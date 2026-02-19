##########################################################################################
# oops/frame/quickframe_.py
##########################################################################################

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from polymath       import Matrix3, Quaternion, Scalar, Vector3
from oops.config    import QUICK, LOGGING
from oops.frame     import Frame
from oops.transform import Transform
import oops.mutable as mutable


class QuickFrame(Frame):
    """QuickFrame is a Frame subclass that returns Transform objects based on
    interpolation of another Frame within a specified time window.
    """

    def __init__(self, frame, tmin, tmax, quick=None):
        """Constructor for a QuickFrame.

        Parameters:
            frame (Frame or str): The Frame or the ID of the Frame that this QuickFrame
                will emulate.
            tmin (float): The earliest time to tabulate in this QuickFrame.
            tmax (float): The latest time to tabulate in this QuickFrame.
            quick (dict, optional): A dictionary containing overrides of any of the
                default values in the default dictionary QUICK.dictionary.
        """

        frame = Frame.as_frame(frame)
        if frame._shape != ():
            raise ValueError('shape of QuickFrame must be ()')
        if isinstance(frame, QuickFrame):
            raise ValueError('QuickFrame cannot be constructed from another QuickFrame')

        mutable.refresh(frame)
        self._slowframe = frame
        self._wayframe  = frame._wayframe
        self._primary   = frame._primary
        self._frame_id  = frame._frame_id
        self._reference = frame._reference
        self._origin    = frame._origin
        self._shape     = ()

        # Expand the time limits a little bit and round them to multiples of tstep
        quickdict = QUICK.dictionary.copy()
        if quick:
            quickdict.update(quick)
        self._quickdict = quickdict

        if not quickdict['use_quickframes']:
            return frame

        tstep = quickdict['frame_time_step']
        extend = quickdict['frame_time_extension']
        extras = int(quickdict['frame_extra_steps'])
        self._input_tmin = tmin
        self._input_tmax = tmax
        self._tstep = tstep
        self._tmin = tstep * ((tmin - extend) // tstep - extras)
        self._tmax = tstep * ((tmax + extend) // tstep + extras + 1)

        self._quickdict = quickdict
        self._omega_numerical = quickdict['quickframe_numerical_omega']
        self._omega_zero = quickdict['ignore_quickframe_omega']
        self._omega_fixed = None        # filled in by first call to `_refresh`

        mutable.refresh(self)

        # Test the precision
        precision = quickdict['frame_self_check']
        if precision is not None:
            time = self._times[:-1] + self._tstep/2.        # halfway points
            (time,
             true_xform) = self._slowframe.transform_at_time_if_possible(time,
                                                                         quick=False)
            (matrix, omega) = self._interpolate_matrix_omega(time)

            # Check largest fractional error
            dmatrix_vals = true_xform.matrix.vals - matrix.vals
            error = np.sqrt(np.sum(dmatrix_vals**2, axis=-1)).max()

            if np.any(true_xform.omega.vals):
                domega = (true_xform.omega - omega).norm() / true_xform.omega.norm()
                error = max(error, domega.max(builtins=True))

            if error > precision:
                raise ValueError(f'precision failure: {error:.3f} > {precision}')

    def _refresh(self):

        times = np.arange(self._tmin, self._tmax + self._tstep/2., self._tstep)
        self._steps = len(times)
        (times, self._xforms) = self._slowframe.transform_at_time_if_possible(times)
        self._times = times.vals

        # Check for omega requirement
        if self._omega_fixed is None:
            self._omega_fixed = not np.any(self._xforms.omega.vals)

        self._spline_setup()

    def _spline_setup(self):
        """Set up the internal tabulation to be interpolated, based on `_times` and
        `_xforms`.
        """

        KIND = 3

        # Create splines for all four components of the quaternion
        quaternions = Quaternion.as_quaternion(self._xforms.matrix)
        self._quat_splines = np.empty((4,), dtype='object')
        for i in range(4):
            self._quat_splines[i] = InterpolatedUnivariateSpline(self._times,
                                                                 quaternions.vals[...,i],
                                                                 k=KIND)

        # Don't interpolate omega if frame is inertial
        if self._omega_zero or self._omega_fixed:
            self._omega_splines = None
            self._qdot_splines = None

        # Create derivative splines if omega solution is numerical
        elif self._omega_numerical:
            self._omega_splines = None
            self._qdot_splines = np.empty((4,), dtype='object')
            for i in range(4):
                self._qdot_splines[i] = self._quat_splines[i].derivative(1)

        # Otherwise, create splines for the vector components of omega
        else:
            self._omega_splines = np.empty((3,), dtype='object')
            self._qdot_splines = None
            for i in range(3):
                self._omega_splines[i] = InterpolatedUnivariateSpline(
                                                self._times,
                                                self._xforms.omega.vals[..., i],
                                                k=KIND)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        if self.pickle_quickframe_details:
            return self.__dict__
        else:
            return (self._slowframe, self._input_tmin, self._input_tmax, self._quickdict)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self.__init__(*state)
        else:
            self.__dict__ = state
        mutable.freeze(self)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or _times. It
                rotates vectors from the reference frame to this frame.

        Notes:
            The time and the Frame object are not required to have the same shape;
            standard rules of broadcasting apply.
        """
        (matrix, omega) = self._interpolate_matrix_omega(time)
        return Transform(matrix, omega, self, self._reference, self._origin)

    def transform_at_time_if_possible(self, time, *, quick=False):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method transform_at_time(), this variant tolerates _times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the _times at which the transform could not be evaluated.

        The default behavior is to assume that all _times are valid. As a result, this
        function calls transform_at_time, but also returns the given time Scalar. This
        behavior is overridden by SpiceFrame, where occasional short gaps in a C-kernel
        can be tolerated as long as a QuickFrame interpolates across them.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input _times given.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """
        xform = self.transform_at_time(time)
        return (time, xform)

    def _interpolate_matrix_omega(self, time, collapse_threshold=None):
        """Use the tabulated splines for a quick evaluation of the transform.

        Parameters:
            time (Scalar, array-like, or float): The time(s) at which to evaluate the
                transform.
            collapse_threshold (float, optional): Use linear interpolation between the
                end points if the time interval is below this value.

        Returns:
            (tuple): (`matrix`, `omega`), where `matrix` is the 3x3 transform matrix and
                `omega` is the rotation vector, if any.
        """

        if collapse_threshold is None:
            collapse_threshold = \
                    self._quickdict['quickframe_linear_interpolation_threshold']

        # `time` can only be a 1-D array in the splines
        time = Scalar.as_scalar(time)
        tflat = time.flatten()
        if tflat.size == 0:
            identity = np.zeros(time.shape + (3, 3))
            identity[..., 0, 0] = 1.
            identity[..., 1, 1] = 1.
            identity[..., 2, 2] = 1.
            matrix = Matrix3(identity, True)
            omega = Vector3(np.ones(time.shape + (3,)), True)
            return (matrix, omega)

        tflat_max = np.max(tflat.vals)
        tflat_min = np.min(tflat.vals)
        time_diff = tflat_max - tflat_min

        # Case 1: A single time
        if time_diff == 0.:
            quat = np.empty((4,))
            quat[0] = self._quat_splines[0](tflat_max)
            quat[1] = self._quat_splines[1](tflat_max)
            quat[2] = self._quat_splines[2](tflat_max)
            quat[3] = self._quat_splines[3](tflat_max)

            quat = Quaternion(quat)

            matrix_vals = np.empty(tflat.shape + (3, 3))
            matrix_vals[..., :, :] = Matrix3.as_matrix3(quat).vals
            matrix = Matrix3(matrix_vals)

            if self._omega_splines is not None:
                om = np.empty((3,))
                om[0] = self._omega_splines[0](tflat_max)
                om[1] = self._omega_splines[1](tflat_max)
                om[2] = self._omega_splines[2](tflat_max)

                omega_vals = np.empty(tflat.shape + (3,))
                omega_vals[...,:] = om[:]
                omega = Vector3(omega_vals)

            elif self._qdot_splines is not None:
                qd = np.empty((4,))
                qd[0] = self._qdot_splines[0](tflat_max)
                qd[1] = self._qdot_splines[1](tflat_max)
                qd[2] = self._qdot_splines[2](tflat_max)
                qd[3] = self._qdot_splines[3](tflat_max)
                qdot = Quaternion(qd)

                omega_vals = np.empty(tflat.shape + (3,))
                omega_vals[..., :] = 2. * (qdot / quat).vals[1:4]
                omega = Vector3(omega_vals)

            elif self._omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self._xforms.omega

        # Case 2: Use linear interpolation for a brief enough time span
        elif time_diff <= collapse_threshold:

            # Create a time scalar just containing the end points
            tflat2 = Scalar([tflat_min, tflat_max])

            quat = np.empty((2, 4))
            quat[:,0] = self._quat_splines[0](tflat2.vals)
            quat[:,1] = self._quat_splines[1](tflat2.vals)
            quat[:,2] = self._quat_splines[2](tflat2.vals)
            quat[:,3] = self._quat_splines[3](tflat2.vals)

            frac = (tflat.vals - tflat_min) / time_diff
            quat = Quaternion(quat[0] + (quat[1] - quat[0]) * frac[..., np.newaxis])
            matrix = Matrix3.as_matrix3(quat)

            if self._omega_splines is not None:
                om = np.empty((2, 3))
                om[:, 0] = self._omega_splines[0](tflat2.vals)
                om[:, 1] = self._omega_splines[1](tflat2.vals)
                om[:, 2] = self._omega_splines[2](tflat2.vals)

                omega = Vector3(om[0] + frac[..., np.newaxis] * (om[1] - om[0]))

            elif self._qdot_splines is not None:
                qd = np.empty((2, 4))
                qd[:, 0] = self._qdot_splines[0](tflat2.vals)
                qd[:, 1] = self._qdot_splines[1](tflat2.vals)
                qd[:, 2] = self._qdot_splines[2](tflat2.vals)
                qd[:, 3] = self._qdot_splines[3](tflat2.vals)
                qd_x2 = 2. * qd

                qdot_x2 = Quaternion(qd_x2[0]
                                     + frac[..., np.newaxis] * (qd_x2[1] - qd_x2[0]))

                omega = (qdot_x2 / quat).to_parts()[1]

            elif self._omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self._xforms.omega

        # Case 3: Use spline evaluation
        else:
            quat = np.empty(tflat.shape + (4,))
            quat[..., 0] = self._quat_splines[0](tflat.vals)
            quat[..., 1] = self._quat_splines[1](tflat.vals)
            quat[..., 2] = self._quat_splines[2](tflat.vals)
            quat[..., 3] = self._quat_splines[3](tflat.vals)

            quat = Quaternion(quat)
            matrix = Matrix3.as_matrix3(quat)

            if self._omega_splines is not None:
                om = np.empty(tflat.shape + (3,))
                om[..., 0] = self._omega_splines[0](tflat.vals)
                om[..., 1] = self._omega_splines[1](tflat.vals)
                om[..., 2] = self._omega_splines[2](tflat.vals)

                omega = Vector3(om)

            elif self._qdot_splines is not None:
                qd = np.empty(tflat.shape + (4,))
                qd[..., 0] = self._qdot_splines[0](tflat.vals)
                qd[..., 1] = self._qdot_splines[1](tflat.vals)
                qd[..., 2] = self._qdot_splines[2](tflat.vals)
                qd[..., 3] = self._qdot_splines[3](tflat.vals)

                qdot = Quaternion(qd)
                omega = 2. * (qdot / quat).to_parts()[1]

            elif self._omega_zero:
                omega = Vector3.ZERO

            else:
                omega = self._xforms.omega

        # Return the matrices and rotation vectors
        matrix = matrix.reshape(time.shape)
        if omega.shape != ():
            omega = omega.reshape(time.shape)

        return (matrix, omega)

    ######################################################################################
    # QuickFrame API
    ######################################################################################

    def extend(self, tmin, tmax):
        """Modify this QuickFrame to accommodate a new, extended time interval.

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
            tmin = self._tstep * ((tmin - extend) // self._tstep - extras)
            t = np.arange(tmin, self._tmin, self._tstep)
            (time0, xform0) = self._slowframe.transform_at_time_if_possible(t)
            count0 = len(time0)
        else:
            count0 = 0

        if tmax > self._tmax:
            tmax = self._tstep * ((tmax + extend) // self._tstep + extras + 1)
            t = np.arange(self._tmax + self._tstep, tmax + self._tstep/2., self._tstep)
            (time1, xform1) = self._slowframe.transform_at_time_if_possible(t)
            count1 = len(time1)
        else:
            count1 = 0

        if count0 + count1 == 0:
            return

        # Allocate the new arrays
        old_size = self._times.size
        new_size = old_size + count0 + count1

        times = np.empty(new_size)
        matrix_vals = np.empty((new_size, 3, 3))
        if self._omega_fixed:
            omega_vals = self._xforms.omega.vals
        else:
            omega_vals = np.empty((new_size, 3))

        # Copy the new arrays
        if count0 > 0:
            times[:count0] = time0.vals
            matrix_vals[:count0] = xform0.matrix.vals
            if not self._omega_fixed:
                omega_vals[:count0] = xform0.omega.vals

        if count1 > 0:
            times[-count1:] = time1.vals
            matrix_vals[-count1:] = xform1.matrix.vals
            if not self._omega_fixed:
                omega_vals[-count1:] = xform1.omega.vals
        else:
            count1 = -new_size      # this makes the indexing below work correctly

        times[count0:-count1] = self._times
        matrix_vals[count0:-count1] = self._xforms.matrix.vals
        if not self._omega_fixed:
            omega_vals[count0:-count1] = self._xforms.omega.vals

        # Generate the new _xforms
        self._times = times
        self._tmin = self._times[0]
        self._tmax = self._times[-1]
        self._xforms = Transform(Matrix3(matrix_vals), Vector3(omega_vals),
                                 self._xforms.frame, self._xforms.reference)

        # Regenerate the splines
        self._spline_setup()

    @staticmethod
    def for_frame(frame, time, *, quick=None):
        """A QuickFrame that approximates this Frame within the given time limits.

        A QuickFrame operates by sampling the given frame and then setting up an
        interpolation grid to evaluate in its place. It can substantially speed up
        performance when the same frame must be evaluated repeatedly many times, e.g., for
        every pixel of an image.

        Parameters:
            frame (Frame): The Frame to be approximated.
            time (Scalar or tuple): The set of times at which the frame is to be
                evaluated. This can simply be a tuple (`tmin`, `tmax`) defining the
                beginning and end times.
            quick (dict or bool, optional): If False, no QuickFrame is created and self is
                returned; if a dictionary, then the values provided override the values in
                the default dictionary QUICK.dictionary, and the merged dictionary is
                used.

        Notes:
            QuickFrames generated by this function are saved as a list inside
            `frame._quickframes`. If a pre-existing QuickFrame that covers the time range
            is found in this list, it is returned rather than constructing a new
            QuickFrame. If a QuickFrame is found in the list that partially covers the
            time range, that QuickFrame is extended to cover the full range and returned.
        """

        if not frame._USE_QUICKFRAMES:
            return frame

        if frame._shape:    # the Frame must be shapeless
            return frame

        # Make sure a QuickFrame has been requested
        if quick is None:
            quick = {}
        if not isinstance(quick, dict):
            return frame

        # Obtain the local QuickFrame dictionary
        quickdict = QUICK.dictionary
        if quick:
            quickdict = quickdict.copy()
            quickdict.update(quick)

        if not quickdict['use_quickframes']:
            return frame

        # Determine the time interval
        time = Scalar.as_scalar(time)
        tmin = time.min(builtins=True)
        tmax = time.max(builtins=True)
        if tmin == Scalar.MASKED:
            return frame

        # Initialize the cache if it is missing
        if not hasattr(frame, '_quickframes'):
            frame._quickframes = []

        # If an existing QuickFrame covers the whole time range, just return it
        for quickframe in frame._quickframes:
            if tmin >= quickframe._tmin and tmax <= quickframe._tmax:
                if LOGGING.quickframe_creation:
                    LOGGING.diagnostic(f'Re-using QuickFrame for {frame}, '
                                       f'{tmin:.3f}, {tmax:.3f})')
                return quickframe

        # This is a quick-and-dirty algorithm to determine whether the use of a QuickFrame
        # is worth the effort relative to using the given Frame.
        #
        # We assume that constructing the QuickFrame carries with it a level of overhead
        # equivalent to _OVERHEAD evaluations.
        #
        # Once constructed, the QuickFrame will be evaluated at least _MIN_EVALUATIONS
        # times, and each evaluation will be _SPEEDUP times faster.
        #
        # If the improvement is less than a factor of _MIN_SAVINGS, we might as well use
        # the original frame.
        #
        # These are WAGs but make sure that, under reasonable circumstances, a QuickFrame
        # is created:
        _OVERHEAD = 200
        _MIN_EVALUATIONS = 1000
        _SPEEDUP = 10.
        _MIN_SAVINGS = 0.2

        # Estimate the number of Frame evaluations needed by the QuickFrame
        tstep = quickdict['frame_time_step']
        extend = quickdict['frame_time_extension']
        extras = int(quickdict['frame_extra_steps'])

        evaluations = max(time.size, _MIN_EVALUATIONS)
        savings_per_evaluation = 1. - 1./_SPEEDUP

        # See if any QuickFrame can be efficiently extended
        for quickframe in frame._quickframes:

            # If there's no overlap, skip it
            if (quickframe._tmin > tmax + tstep) or (quickframe._tmax < tmin - tstep):
                continue

            # Otherwise, check the effort involved
            new_duration = (max(tmax, quickframe._tmax) - min(tmin, quickframe._tmin)
                            + 2*extend)
            new_steps = (new_duration/tstep + 2*extras) - quickframe._steps
            overhead = _OVERHEAD + new_steps
            if savings_per_evaluation - overhead/evaluations >= _MIN_SAVINGS:
                if LOGGING.quickframe_creation:
                    LOGGING.diagnostic(f'Extending QuickFrame for {frame}, '
                                       f'{tmin:.3f}, {tmax:.3f})')
                quickframe.extend((tmin, tmax))
                return quickframe

        # Otherwise, construct a new QuickFrame
        steps = (tmax - tmin + 2*extend)/tstep + 2*extras
        overhead = _OVERHEAD + steps
        if savings_per_evaluation - overhead/evaluations >= _MIN_SAVINGS:
            if LOGGING.quickframe_creation:
                LOGGING.diagnostic(f'New QuickFrame for {frame}: {tmin:.3f}, {tmax:.3f})')

            result = QuickFrame(frame, tmin, tmax, quick=quickdict)
            if len(frame._quickframes) >= quickdict['quickframe_cache_size']:
                frame._quickframes = [result] + frame._quickframes[:-1]
            else:
                frame._quickframes = [result] + frame._quickframes
            return result

        return frame

##########################################################################################

Frame._FRAME_SUBCLASSES.append(QuickFrame)
Frame._QuickFrame = QuickFrame

##########################################################################################
