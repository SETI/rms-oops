##########################################################################################
# oops/frame/spiceframe.py: Subclass SpiceFrame of class Frame
##########################################################################################

import numpy as np
from scipy.interpolate import UnivariateSpline

import cspyce

from polymath       import Matrix3, Quaternion, Scalar, Vector3
from oops.frame     import Frame
from oops.transform import Transform
import oops.spice_support as spice


class SpiceFrame(Frame):
    """A Frame defined within the SPICE toolkit."""

    def __init__(self, spice_frame, spice_reference='J2000', frame_id=None, *,
                 omega_type='tabulated', omega_dt=1.):
        """Constructor for a SpiceFrame.

        Parameters:
            spice_frame (str or int): The name, frame ID, or body ID as used in the SPICE
                toolkit.
            spice_reference (str or int, optional): The name or ID of the reference frame
                as used in the SPICE toolkit.
            frame_id (str, optional): The name under which the frame will be registered.
                By default, this is the name as used by the SPICE toolkit.
            omega_type (str, optional): Options defining how `omega`, the time derivative
                of the frame, is calculated:

                * "tabulated" to take `omega` directly from the SPICE kernel;
                * "numerical" to derive `omega` via numerical derivatives;
                * "zero" to ignore omega vectors.

            omega_dt (float, optional): The default time step in seconds to use when
                `omega_type` equals "numerical".
        """

        # Preserve the inputs
        self.spice_frame = spice_frame
        self.spice_reference = spice_reference
        self.omega_type = omega_type
        self.omega_dt = omega_dt

        # Interpret the SPICE frame and reference IDs
        (self.spice_frame_id,
         self.spice_frame_name) = spice.frame_id_and_name(spice_frame)

        (self.spice_reference_id,
         self.spice_reference_name) = spice.frame_id_and_name(spice_reference)

        # Fill in the Frame ID and save it in the SPICE global dictionary
        self.frame_id = frame_id or self.spice_frame_name
        spice.FRAME_TRANSLATION[self.spice_frame_id] = self.frame_id
        spice.FRAME_TRANSLATION[self.spice_frame_name] = self.frame_id

        # Fill in the reference wayframe
        reference_id = spice.FRAME_TRANSLATION[self.spice_reference_id]
        self.reference = Frame.as_wayframe(reference_id)

        # Fill in the origin waypoint
        self.spice_origin_id = cspyce.frinfo(self.spice_frame_id)[0]
        self.spice_origin_name = cspyce.bodc2n(self.spice_origin_id)
        origin_id = spice.PATH_TRANSLATION[self.spice_origin_id]

        try:
            self.origin = Frame.PATH_CLASS.as_waypoint(origin_id)
        except KeyError:
            # If the origin path was never defined, define it now
            origin_path = Frame.SPICEPATH_CLASS(origin_id)
            self.origin = origin_path.waypoint

        # No shape
        self.shape = ()

        # Save interpolation method
        if omega_type not in ('tabulated', 'numerical', 'zero'):
            raise ValueError('invalid SpiceFrame omega_type: ' + repr(omega_type))

        self.omega_tabulated = (omega_type == 'tabulated')
        self.omega_numerical = (omega_type == 'numerical')
        self.omega_zero = (omega_type == 'zero')

        # Always register a SpiceFrame. This also fills in the waypoint.
        self.register()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self.spice_frame_name, self.spice_reference, self.omega_type,
                self.omega_dt, self._state_id())

    def __setstate__(self, state):
        (spice_frame_name, spice_reference, omega_type, omega_dt, frame_id) = state
        if frame_id is None:
            frame_id = spice.FRAME_TRANSLATION.get(spice_frame_name, None)
        self.__init__(spice_frame_name, spice_reference, frame_id=frame_id,
                      omega_type=omega_type, omega_dt=omega_dt)

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, quick={}):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Transform): The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.

        """

        time = Scalar.as_scalar(time).as_float()

        # Handle a single time
        if time.shape == ():

            # Case 1: omega_type = tabulated
            if self.omega_tabulated:
                matrix6 = cspyce.sxform(self.spice_reference_name, self.spice_frame_name,
                                        time.values)
                (matrix, omega) = cspyce.xf2rav(matrix6)
                return Transform(matrix, omega, self, self.reference)

            # Case 2: omega_type = zero
            elif self.omega_zero:
                matrix = cspyce.pxform(self.spice_reference_name, self.spice_frame_name,
                                       time.values)

                return Transform(matrix, Vector3.ZERO, self, self.reference)

            # Case 3: omega_type = numerical
            else:
                et = time.vals
                times = np.array((et - self.omega_dt, et, et + self.omega_dt))
                mats = np.empty((3,3,3))

                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name, times[j])

                # Convert three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2, s=0)
                    qdot[j] = spline.derivative(1)(et)

                omega = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                return Transform(mats[1], omega, self, self.reference)

        # Apply the quick_frame if requested

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self.omega_numerical
            quick['ignore_quickframe_omega'] = self.omega_zero

            if self.omega_numerical:
                quick['frame_time_step'] = min(quick['frame_time_step'], self.omega_dt)

            frame = self.quick_frame(time, quick)
            return frame.transform_at_time(time, quick=False)

        # Handle multiple times

        # Case 1: omega_type = tabulated
        if self.omega_tabulated:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                matrix6 = cspyce.sxform(self.spice_reference_name,
                                        self.spice_frame_name, t)
                (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

        # Case 2: omega_type = zero
        elif self.omega_zero:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.zeros(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                matrix[i] = cspyce.pxform(self.spice_reference_name,
                                          self.spice_frame_name, t)

        # Case 3: omega_type = numerical
        # This procedure calculates each omega using its own UnivariateSpline; it could be
        # very slow. A QuickFrame is recommended as it would accomplish the same goals
        # much faster.
        else:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):

                # Define a set of three times centered on given time
                times = np.array((t - self.omega_dt, t, t + self.omega_dt))

                # Generate the rotation matrix at each time
                mats = np.empty((3,3,3))
                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name, times[j])

                # Convert these three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2, s=0)
                    qdot[j] = spline.derivative(1)(t)

                omega[i] = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                matrix[i] = mats[1]

        return Transform(matrix, omega, self, self.reference)

    def transform_at_time_if_possible(self, time, quick={}):
        """Transform that rotates coordinates from the reference frame to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Unlike method `transform_at_time`, this variant tolerates times that raise cspyce
        errors. It returns a new time Scalar along with the new Transform, where both
        objects skip over the times at which the transform could not be evaluated.

        Parameters:
            time (Scalar, array-like, or float): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (tuple): A tuple with two values:

            * Scalar: The times that are returned, possibly containing a subset of the
                original times given.
            * Transform: The Tranform applicable at the specified time or times. It
                rotates vectors from the reference frame to this frame.
        """

        time = Scalar.as_scalar(time).as_float()

        # A single input time can be handled via the previous method
        if time.shape == ():
            return (time, self.transform_at_time(time, quick))

        # Apply the quick_frame if requested

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self.omega_numerical
            quick['ignore_quickframe_omega'] = self.omega_zero

            if self.omega_numerical:
                quick['frame_time_step'] = min(quick['frame_time_step'], self.omega_dt)

            frame = self.quick_frame(time, quick)
            return frame.transform_at_time_if_possible(time, quick=False)

        # Handle multiple times

        # Lists used in case of error
        new_time = []
        matrix_list = []
        omega_list = []

        error_found = None

        # Case 1: omega_type = tabulated
        if self.omega_tabulated:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.empty(time.shape + (3,))

            for i, t in np.ndenumerate(time.values):
                try:
                    matrix6 = cspyce.sxform(self.spice_reference_name,
                                            self.spice_frame_name, t)
                    (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

                    new_time.append(t)
                    matrix_list.append(matrix[i])
                    omega_list.append(omega[i])

                except (RuntimeError, ValueError, IOError) as e:
                    if len(time.shape) > 1:
                        raise e
                    error_found = e

        # Case 2: omega_type = zero
        elif self.omega_zero:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.zeros(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
                try:
                    matrix[i] = cspyce.pxform(self.spice_reference_name,
                                              self.spice_frame_name, t)

                    new_time.append(t)
                    matrix_list.append(matrix[i])
                    omega_list.append((0., 0., 0.))

                except (RuntimeError, ValueError, IOError) as e:
                    if len(time.shape) > 1:
                        raise e
                    error_found = e

        # Case 3: omega_type = numerical
        # This procedure calculates each omega using its own UnivariateSpline; it could be
        # very slow. A QuickFrame is recommended as it would accomplish the same goals
        # much faster.
        else:
            matrix = np.empty(time.shape + (3, 3))
            omega  = np.empty(time.shape + (3,))

            for i,t in np.ndenumerate(time.values):
              try:
                times = np.array((t - self.omega_dt, t, t + self.omega_dt))
                mats = np.empty((3, 3, 3))

                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self.spice_reference_name,
                                            self.spice_frame_name, times[j])

                # Convert three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.values[:,j], k=2, s=0)
                    qdot[j] = spline.derivative(1)(t)

                omega[i] = 2. * (Quaternion(qdot) / quats[1]).values[1:4]
                matrix[i] = mats[1]

                new_time.append(t)
                matrix_list.append(matrix[i])
                omega_list.append(omega[i])

              except (RuntimeError, ValueError, IOError) as e:
                if len(time.shape) > 1:
                    raise e
                error_found = e

        if error_found is not None:
            if len(new_time) == 0:
                raise error_found

            time = Scalar(new_time)
            matrix = Matrix3(matrix_list)
            omega = Vector3(omega_list)
        else:
            matrix = Matrix3(matrix)
            omega = Vector3(omega)

        return (time, Transform(matrix, omega, self, self.reference))

##########################################################################################
