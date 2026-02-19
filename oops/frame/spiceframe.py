##########################################################################################
# oops/frame/spiceframe.py: Subclass SpiceFrame of class Frame
##########################################################################################

import numbers
import numpy as np
from scipy.interpolate import UnivariateSpline

import cspyce

from polymath              import Matrix3, Quaternion, Scalar, Vector3
from oops.frame            import Frame, J2000Frame, LinkedFrame, NullFrame
from oops.frame.quickframe import QuickFrame
from oops.transform        import Transform


class SpiceFrame(Frame):
    """A Frame defined within the SPICE toolkit."""

    _WAYFRAMES = {}         # frame_key -> wayframe
    _FOR_NAME = {}          # SPICE frame name -> first defined SpiceFrame
    _FRAME_LOOKUP = {}      # (name, reference name, omega_type, omega_dt) -> SpiceFrame

    _USE_QUICKFRAMES = True     # Overrides default to enable QuickFrames

    def __init__(self, spice_frame, reference=None, *, omega_type='tabulated',
                 omega_dt=1., frame_id=None):
        """Constructor for a SpiceFrame.

        Parameters:
            spice_frame (str or int): The name, frame code, or body code as used in the
                SPICE toolkit.
            reference (SpiceFrame or str, optional): The Frame or ID of the Frame relative
                to which this frame is defined. This must be a SpiceFrame or else, by
                default, J2000.
            omega_type (str, optional): Options defining how `omega`, the time derivative
                of the frame, is calculated:

                * "tabulated" to take `omega` directly from the SPICE kernel (default);
                * "numerical" to derive `omega` via numerical derivatives;
                * "zero" to ignore omega vectors. This is the default for inertial frames.

            omega_dt (float, optional): The default time step in seconds to use when
                `omega_type` equals "numerical"; default is 1.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceFrames are always registered.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name, frame code, body
                name, or body code within the SPICE Toolkit.
            ValueError: If `reference` is not a SpiceFrame or J2000, or if `omega_type`
                does not have a recognized value.
        """

        self._fill_spice_info(spice_frame, reference)

        self._is_inertial = cspyce.frinfo(self._spice_frame_name)[1] == 1   # frame class
        self._reference_is_inertial = cspyce.frinfo(self._spice_reference_name)[1] == 1
        if self._is_inertial and self._reference_is_inertial:
            omega_type = 'zero'

        # Handle the omega parameters
        self._omega_type = omega_type or 'tabulated'
        self._omega_dt = float(omega_dt or 1.)
        self._omega_tabulated = (omega_type == 'tabulated')
        self._omega_numerical = (omega_type == 'numerical')
        self._omega_zero = (omega_type == 'zero')
        if self._omega_type not in {'tabulated', 'numerical', 'zero'}:
            raise ValueError(f'invalid SpiceFrame omega_type: {self._omega_type}')

        # If the reference is J2000, register as normal
        if self._reference == Frame.J2000:
            _ = SpiceFrame._FOR_NAME.setdefault(self._spice_frame_name, self)
            self._register(frame_id or self._spice_frame_name.replace(' ', '_'))
        else:
            # Otherwise, construct the primary version first
            wrt_j2000 = SpiceFrame.get(self._spice_frame_name, Frame.J2000,
                                       omega_type=omega_type, omega_dt=omega_dt,
                                       frame_id=frame_id)
            # Cache but don't register under this frame ID
            self._register(frame_id=None)
            self._wayframe = wrt_j2000._wayframe
            self._frame_id = wrt_j2000._frame_id

    def _fill_spice_info(self, spice_frame, reference):
        """Fill in this object's spice codes and names, plus the origin and reference.

        Used by both SpiceFrame and SpiceType1Frame.
        """

        # Interpret the SPICE frame
        (self._spice_frame_code,
         self._spice_frame_name) = SpiceFrame._frame_code_and_name(spice_frame)

        # Determine the reference frame
        self._reference = reference and Frame.as_wayframe(reference) or Frame.J2000
        if self._reference == Frame.J2000:
            self._spice_reference_code = 1
            self._spice_reference_name = 'J2000'
        elif isinstance(self._reference, SpiceFrame):
            self._spice_reference_code = self._reference._spice_frame_code
            self._spice_reference_name = self._reference._spice_frame_name
        else:
            raise ValueError(f'{type(self).__name__} reference must be a SpiceFrame or '
                             'J2000')

        # Determine the origin Path, constructing it if necessary
        spice_origin_code = cspyce.frinfo(self._spice_frame_name)[0]
        if spice_origin_code == 0:  # if the origin is the SSB, this Frame is inertial
            self._origin = None
            self._is_inertial = True
            if self._reference._origin is None:     # no omega if both frames are inertial
                omega_type = 'zero'
        else:
            self._origin = Frame._SpicePath.get(spice_origin_code)
            self._is_inertial = False

        self._shape = ()

    def _refresh(self):
        if hasattr(self, '_quickframes'):
            self._quickframes.clear()

    def _wayframe_key(self):
        return self._spice_frame_name

    @staticmethod
    def _frame_code_and_name(arg):
        """The spice_code and spice_name of frame in the SPICE Toolkit given a code or
        name.
        """

        # Interpret an integer input
        if isinstance(arg, numbers.Integral):
            try:
                name = cspyce.frmnam_error(arg)
            except (KeyError, LookupError):
                pass
            else:
                return (arg, name)

            # Otherwise, perhaps it is a body code
            if not cspyce.bodfnd(arg, 'POLE_RA'):
                raise LookupError(f'unrecognized SPICE frame {arg}')
            try:
                return tuple(cspyce.cidfrm(arg))
            except (KeyError, LookupError):
                raise LookupError(f'unrecognized SPICE frame {arg}')

        # Interpret a string input
        else:
            # Validate this as the name of a frame
            try:
                frame_code = cspyce.namfrm_error(arg)
            except (KeyError, LookupError):
                pass
            else:
                # Make sure the frame is defined
                body_code = cspyce.frinfo(frame_code)[0]
                if body_code > 0 and not cspyce.bodfnd(body_code, 'POLE_RA'):
                    raise LookupError(f'frame "{arg}" is undefined')
                return (frame_code, cspyce.frmnam(frame_code))

            # See if this is the name of a body
            try:
                body_code = cspyce.bodn2c_error(arg)
            except (KeyError, LookupError):
                raise LookupError(f'unrecognized SPICE frame "{arg}"')

            # Make sure the body's frame is defined
            if not cspyce.bodfnd(body_code, 'POLE_RA'):
                raise LookupError(f'frame for body "{arg}" is undefined')

            # Return the name of the associated frame
            return tuple(cspyce.cidfrm(body_code))

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self._spice_frame_name, self._reference, self._omega_type, self._omega_dt,
                self.stripped_id, self._get_quickframes())

    def __setstate__(self, state):
        (frame_name, reference, omega_type, omega_dt, frame_id, quickframes) = state
        self.__init__(frame_name, reference, omega_type=omega_type, omega_dt=omega_dt,
                      frame_id=frame_id)
        if quickframes:
            self._quickframes = quickframes

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
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
            if self._omega_tabulated:
                matrix6 = cspyce.sxform(self._spice_reference_name,
                                        self._spice_frame_name, time.vals)
                (matrix, omega) = cspyce.xf2rav(matrix6)
                return Transform(matrix, omega, self, self.reference)

            # Case 2: omega_type = zero
            elif self._omega_zero:
                matrix = cspyce.pxform(self._spice_reference_name, self._spice_frame_name,
                                       time.vals)
                return Transform(matrix, Vector3.ZERO, self, self.reference)

            # Case 3: omega_type = numerical
            else:
                et = time.vals
                times = np.array((et - self._omega_dt, et, et + self._omega_dt))
                mats = cspyce.pxform_vector(self._spice_reference_name,
                                            self._spice_frame_name, times)

                # Convert three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.vals[:,j], k=2, s=0)
                    qdot[j] = spline.derivative(1)(et)

                omega = 2. * (Quaternion(qdot) / quats[1]).vals[1:4]
                return Transform(mats[1], omega, self, self._reference)

        # Use a QuickFrame if warranted
        if quick is None:
            quick = {}

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self._omega_numerical
            quick['ignore_quickframe_omega'] = self._omega_zero

            if self._omega_numerical:
                quick['frame_time_step'] = min(quick.get('frame_time_step', np.inf),
                                               self._omega_dt)

            frame = self.quick_frame(time, quick=quick)
            if isinstance(frame, QuickFrame):
                return frame.transform_at_time(time, quick=False)

        # Handle multiple times
        matrix = np.empty(time.shape + (3, 3))
        omega  = np.zeros(time.shape + (3,))

        # Case 1: omega_type = tabulated
        if self._omega_tabulated:
            for i, t in np.ndenumerate(time.vals):
                matrix6 = cspyce.sxform(self._spice_reference_name,
                                        self._spice_frame_name, t)
                (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

        # Case 2: omega_type = zero
        elif self._omega_zero:
            for i,t in np.ndenumerate(time.vals):
                matrix[i] = cspyce.pxform(self._spice_reference_name,
                                          self._spice_frame_name, t)

        # Case 3: omega_type = numerical
        # This procedure calculates each omega using its own UnivariateSpline; it could be
        # very slow. A QuickFrame is recommended as it would accomplish the same goals
        # much faster.
        else:
            for i, t in np.ndenumerate(time.vals):

                # Define a set of three times centered on given time
                times = np.array((t - self._omega_dt, t, t + self._omega_dt))

                # Generate the rotation matrix at each time
                mats = np.empty((3, 3, 3))
                for j in range(len(times)):
                    mats[j] = cspyce.pxform(self._spice_reference_name,
                                            self._spice_frame_name, times[j])

                # Convert these three matrices to quaternions
                quats = Quaternion.as_quaternion(Matrix3(mats))

                # Use a Univariate spline to get components of the derivative
                qdot = np.empty(4)
                for j in range(4):
                    spline = UnivariateSpline(times, quats.vals[:, j], k=2, s=0)
                    qdot[j] = spline.derivative(1)(t)

                omega[i] = 2. * (Quaternion(qdot) / quats[1]).vals[1:4]
                matrix[i] = mats[1]

        matrix = Matrix3(matrix, mask=time.mask)
        return Transform(matrix, omega, self, self._reference)

    def transform_at_time_if_possible(self, time, *, quick=None):
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
            (tuple): The tuple (`newtimes`, `transform`), where:

            * `newtimes` (Scalar): Times at which `transform` has been provided; this may
              be a subset of the input times given because it omits times at which the
              Transform could not be evaluated.
            * `transform` (Transform): The Tranform applicable at `newtimmes`. It rotates
              vectors from the reference frame to this frame.
        """

        time = Scalar.as_scalar(time).as_float()

        # A single input time can be handled via the previous method
        if time.shape == ():
            return (time, self.transform_at_time(time, quick=quick))

        # Apply the QuickFrame if requested
        if quick is None:
            quick = {}

        if isinstance(quick, dict):
            quick = quick.copy()
            quick['quickframe_numerical_omega'] = self._omega_numerical
            quick['ignore_quickframe_omega'] = self._omega_zero

            if self._omega_numerical:
                quick['frame_time_step'] = min(quick.get('frame_time_step', np.inf),
                                               self._omega_dt)

            frame = self.quick_frame(time, quick=quick)
            return frame.transform_at_time_if_possible(time, quick=False)

        # Handle multiple times
        matrix = np.empty(time.shape + (3, 3))
        omega  = np.zeros(time.shape + (3,))

        # Lists used in case of error
        new_time = []
        matrix_list = []
        omega_list = []

        error_found = None

        # Case 1: omega_type = tabulated
        if self._omega_tabulated:
            for i, t in np.ndenumerate(time.vals):
                try:
                    matrix6 = cspyce.sxform(self._spice_reference_name,
                                            self._spice_frame_name, t)
                    (matrix[i], omega[i]) = cspyce.xf2rav(matrix6)

                    new_time.append(t)
                    matrix_list.append(matrix[i])
                    omega_list.append(omega[i])

                except (RuntimeError, ValueError, IOError) as e:
                    if len(time.shape) > 1:
                        raise e
                    error_found = e

        # Case 2: omega_type = zero
        elif self._omega_zero:
            for i, t in np.ndenumerate(time.vals):
                try:
                    matrix[i] = cspyce.pxform(self._spice_reference_name,
                                              self._spice_frame_name, t)

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
            for i, t in np.ndenumerate(time.vals):
                try:
                    times = np.array((t - self._omega_dt, t, t + self._omega_dt))
                    mats = np.empty((3, 3, 3))

                    for j in range(len(times)):
                        mats[j] = cspyce.pxform(self._spice_reference_name,
                                                self._spice_frame_name, times[j])

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

    ######################################################################################
    # SpiceFrame API
    ######################################################################################

    @staticmethod
    def get(spice_frame, reference=None, *, omega_type='tabulated', omega_dt=1.,
            frame_id=None):
        """The SpiceFrame defined by the given parameters.

        If a matching SpiceFrame already exists, it is returned; otherwise, a new one is
        constructed and returned.

        Parameters:
            spice_frame (str, int, or SpiceFrame): The frame name, frame code, body name,
                or body code as used in the SPICE toolkit. Alternatively, an existing
                SpiceFrame (which might use the wrong reference frame).
            reference (SpiceFrame or str, optional): The SpiceFrame or frame ID relative
                to which this frame refers. This must be a SpiceFrame or else, by default,
                J2000.
            omega_type (str, optional): Options defining how `omega`, the time derivative
                of the frame, is calculated:

                * "tabulated" to take `omega` directly from the SPICE kernel;
                * "numerical" to derive `omega` via numerical derivatives;
                * "zero" to ignore omega vectors. This is the default for inertial frames.

            omega_dt (float, optional): The default time step in seconds to use when
                `omega_type` equals "numerical". By default, the `omega_dt` of the
                returned SpiceFrame is not constrained.
            frame_id (str, optional): The ID under which to register this Frame. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpiceFrames are always registered. This input is used only if a new
                SpiceFrame is constructed; otherwise, the pre-existing ID is retained.

        Returns:
            (SpiceFrame): The SpiceFrame, newly constructed if necessary.

        Raises:
            LookupError: If `spice_frame` is not a recognized frame name, frame code, body
                name, or body code within the SPICE Toolkit.
        """

        reference = Frame.as_wayframe(reference)

        # Handle a SpiceFrame input; use it if it matches
        if isinstance(spice_frame, SpiceFrame):
            if (reference == spice_frame._reference
                    and omega_type == spice_frame._omega_type
                    and (omega_dt == spice_frame._omega_dt
                         or not spice_frame._omega_numerical)):
                return spice_frame
            # Otherwise, identify the name and continue
            name = spice_frame._spice_frame_name
        else:
            (_, name) = SpiceFrame._frame_code_and_name(spice_frame)

        # See if a pre-existing Frame matches the request (including omega options)
        if name == reference._spice_frame_name:
            if name == 'J2000':
                return Frame.J2000
            else:
                return NullFrame(reference)

        key = (name, reference._spice_frame_name, omega_type, omega_dt)
        if key in SpiceFrame._FRAME_LOOKUP:
            return SpiceFrame._FRAME_LOOKUP[key]

        # Otherwise, we need a new SpiceFrame
        return SpiceFrame(name, reference, omega_type=omega_type, omega_dt=omega_dt,
                          frame_id=frame_id)

    def _get_shortcut(self, reference):
        """A Frame that directly transforms from the given reference to this SpiceFrame.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between SpiceFrames very efficiently.

        Parameters:
            reference (Frame): The reference Frame, which must be a valid wayframe.
        """

        # Find the first SpiceFrame (or J2000) that's an ancestor of the reference
        ancestor = reference
        while not isinstance(ancestor, (SpiceFrame, J2000Frame)):
            ancestor = ancestor._reference

        # Get the SpiceFrame to the selected ancestor
        spice_frame = SpiceFrame.get(self, ancestor, omega_type=self._omega_type,
                                     omega_dt=self._omega_dt)

        # Maybe we're done
        if ancestor == reference:
            return spice_frame

        # Get the "remainder" frame from the ancestor to the reference, then link
        remainder = ancestor._wrt(reference, use_shortcuts=False)
        return LinkedFrame(spice_frame, remainder)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(SpiceFrame)
Frame._SpiceFrame = SpiceFrame

##########################################################################################
