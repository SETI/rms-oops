##########################################################################################
# oops/path/spicepath.py
##########################################################################################

import numbers
import numpy as np

import cspyce

from polymath              import Scalar
from oops.event            import Event
from oops.frame.frame_     import Frame, J2000Frame
from oops.frame.spiceframe import SpiceFrame
from oops.path.path_       import Path, LinkedPath, SSBPath
from oops.path.quickpath   import QuickPath


class SpicePath(Path):
    """A Path subclass that returns information based on a SPICE SP kernel."""

    _WAYPOINTS = {}
    _FOR_CODE = {}              # SPICE path code -> SpicePath
    _USE_QUICKPATHS = True      # Overrides default to enable QuickPaths

    def __init__(self, spice_path, origin=None, frame=None, *, path_id=None):
        """Constructor for a SpicePath.

        Parameters:
            spice_path (str or int): The SPICE toolkit identification of the target body
                as a name or integer.
            origin (SpicePath or str, optional): The Path or the ID of the Path relative
                to which this Path is defined. This must be a SpicePath or else, by
                default, the Solar System Barycenter.
            frame (SpiceFrame or str, optional): The Frame or the ID of the Frame for the
                returned Path coordinates. This must be a SpiceFrame or else, by default,
                J2000.
            path_id (str, optional): The ID under which to register this Path. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpicePaths are always registered.

        Raises:
            LookupError: If `spice_path` is not a recognized body name or code within the
                SPICE Toolkit.
            ValueError: If `origin` is not a SpicePath or `frame` is not a SpiceFrame.
        """

        # Interpret the SPICE path
        (self._spice_path_code,
         self._spice_path_name) = SpicePath._body_code_and_name(spice_path)

        # Fill in the origin info
        self._origin = Path.as_waypoint(origin)
        if not isinstance(self._origin, (SpicePath, SSBPath)):
            raise ValueError('SpicePath origin must be a SpicePath or SSB')
        if self._origin == Path.SSB:
            self._spice_origin_code = 0
            self._spice_origin_name = 'SSB'
        else:
            self._spice_origin_code = self._origin._spice_path_code
            self._spice_origin_name = self._origin._spice_path_name

        # Fill in the frame info
        self._frame = Frame.as_wayframe(frame)
        if not isinstance(self._frame, (SpiceFrame, J2000Frame)):
            raise ValueError('SpicePath frame must be a SpiceFrame or J2000')
        if self._frame == Frame.J2000:
            self._spice_frame_code = 1
            self._spice_frame_name = 'J2000'
        else:
            self._spice_frame_code = self._frame._spice_frame_code
            self._spice_frame_name = self._frame._spice_frame_name

        self._shape = ()

        # If the reference is not SSB/J2000, construct the primary version first
        if self._origin != Path.SSB or self._frame != Frame.J2000:
            wrt_ssb = SpicePath.get(self._spice_path_code, Path.SSB, Frame.J2000,
                                    path_id=path_id)
            # Cache but don't register under this path ID
            self._register(path_id=None)
            self._waypoint = wrt_ssb._waypoint
            self._path_id = wrt_ssb._path_id
        else:
            _ = SpicePath._FOR_CODE.setdefault(self._spice_path_code, self)
            self._register(path_id or self._spice_path_name.replace(' ', '_'))

        # Also make this Path findable under the name the caller used
        if isinstance(spice_path, str):
            SpicePath._register_alias(spice_path, self)

    def _refresh(self):
        if hasattr(self, '_quickpaths'):
            self._quickpaths.clear()

    def _waypoint_key(self):
        return self._spice_path_code

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        if self._origin == Path.SSB and self._frame == Frame.J2000:
            return f'{type(self).__name__}("{self._spice_path_name}")'

        parts = [f'{type(self).__name__}("{self._spice_path_name}"'
                 f'{blanks}{self._origin.show(level-1, skip)}']
        if self._frame != Frame.J2000:
            parts.append(f'{blanks}{self._frame.show(level-1, skip)}')
        return ',\n'.join(parts) + ')'

    @staticmethod
    def _body_code_and_name(arg):
        """The SPICE body code and body name of a body, given either a code or a name.

        Parameters:
            arg (str or int): The SPICE toolkit identification of the body as a name or
                integer.

        Returns:
            tuple[int, str]: The body's SPICE code and its name as defined in the SPICE
            Toolkit.

        Raises:
            LookupError: If `arg` is not a recognized body name or code within the SPICE
                Toolkit.
        """

        # Interpret an integer input
        if isinstance(arg, numbers.Integral):
            try:
                name = cspyce.bodc2n_error(arg)
            except (KeyError, LookupError):
                raise LookupError(f'unrecognized SPICE body {arg}')

            return (arg, name)

        # Interpret a string input
        else:
            try:
                body_code = cspyce.bodn2c_error(arg)
                name = cspyce.bodc2n_error(body_code)
            except (KeyError, LookupError):
                raise LookupError(f'unrecognized SPICE body "{arg}"')

            return (body_code, name)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        return (self._spice_path_code, self._origin, self._frame, self.stripped_id)

    def __setstate__(self, state):
        (spice_path_code, origin, frame, path_id) = state
        self.__init__(spice_path_code, origin, frame, path_id=path_id)

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Event: The Event object containing (at least) the time, position, and velocity
            on this Path.
        """

        time = Scalar.as_scalar(time).as_float()

        # A single time can be handled quickly
        if time.shape == ():
            (state, lighttime) = cspyce.spkez(self._spice_path_code,
                                              time.vals,
                                              self._spice_frame_name,
                                              'NONE',
                                              self._spice_origin_code)
            return Event(time, (state[0:3], state[3:6]), self._origin, self._frame)

        # Use a QuickPath if warranted
        if quick is None:
            quick = {}
        if isinstance(quick, dict):
            path = self.quick_path(time, quick=quick)
            if isinstance(path, QuickPath):
                return path.event_at_time(time, quick=False)

        # Handle multiple times

        # Fill in the states and light travel times using cspyce
        if np.any(time.mask):
            state = cspyce.spkez_vector(self._spice_path_code,
                                        time.vals[time.antimask],
                                        self._spice_frame_name,
                                        'NONE',
                                        self._spice_origin_code)[0]
            pos = np.zeros(time.shape + (3,))
            vel = np.zeros(time.shape + (3,))
            pos[time.antimask] = state[..., 0:3]
            vel[time.antimask] = state[..., 3:6]

        else:
            state = cspyce.spkez_vector(self._spice_path_code,
                                        time.vals.ravel(),
                                        self._spice_frame_name,
                                        'NONE',
                                        self._spice_origin_code)[0]
            pos = state[:, 0:3].reshape(time.shape + (3,))
            vel = state[:, 3:6].reshape(time.shape + (3,))

        # Convert to an Event and return
        return Event(time, (pos, vel), self._origin, self._frame)

    ######################################################################################
    # SpicePath API
    ######################################################################################

    @staticmethod
    def _register_alias(alias, path):
        """Register an additional ID for a Path, leaving its primary ID unchanged.

        The SPICE Toolkit recognizes several names for the same body; "GLL" and "GALILEO
        ORBITER" both refer to body -77. A caller that identifies a body by one of these
        names expects to look the Path up again by that same name, so the name used is
        registered alongside the canonical one.

        An alias already present in the registry is left alone, so this can never take an
        ID away from the Path that owns it.

        Parameters:
            alias (str): An additional ID under which to register the Path.
            path (Path): The Path to be found under this ID.
        """

        if not alias or alias in Path._PATH_REGISTRY:
            return

        # Resolve through the primary ID so that both IDs return the same object
        Path._PATH_REGISTRY[alias] = Path._PATH_REGISTRY.get(path._path_id, path)

    @staticmethod
    def get(spice_path, origin=None, frame=None, *, path_id=None):
        """The SpicePath defined by the given parameters.

        If a matching SpicePath already exists, it is returned; otherwise, a new SpicePath
        is constructed and returned.

        Parameters:
            spice_path (str or int): The SPICE toolkit identification of the target body
                as a name or integer.
            origin (SpicePath or str, optional): The Path or the ID of the Path relative
                to which this Path is defined. This must be a SpicePath or else, by
                default, the Solar System Barycenter.
            frame (SpiceFrame or str, optional): The Frame or the ID of the Frame for the
                returned Path coordinates. This must be a SpiceFrame or else, by default,
                J2000.
            path_id (str, optional): The ID under which to register this Path. If not
                specified, the name as defined in the SPICE Toolkit is used. Note that
                SpicePaths are always registered. This input is used only if a new
                SpicePath is constructed; otherwise, the pre-existing ID is retained.

        Returns:
            SpicePath: The SpicePath, newly constructed if necessary.

        Raises:
            LookupError: If `spice_path` is not a recognized body name or code within the
                SPICE Toolkit.
        """

        origin = Path.as_waypoint(origin)
        frame = Frame.as_wayframe(frame)

        # Handle a SpicePath input; use it if it matches
        if isinstance(spice_path, SpicePath):
            if origin == spice_path._origin and frame == spice_path._frame:
                return spice_path
            # Otherwise, identify the code and continue
            code = spice_path._spice_path_code
        else:
            (code, _) = SpicePath._body_code_and_name(spice_path)

        # Intervene for the SSB, but only where the canonical SSB Path is what was asked
        # for. Relative to any other origin or frame, the SSB needs a Path of its own;
        # returning Path.SSB here would silently ignore `origin` and `frame`.
        if code == 0 and origin == Path.SSB and frame == Frame.J2000:
            result = Path.SSB

        # If this body code has not been used, construct a new SpicePath
        elif code not in SpicePath._FOR_CODE:
            result = SpicePath(code, origin, frame, path_id=path_id)

        else:
            # Use the Path we need if it is already registered
            waypoint = SpicePath._FOR_CODE[code]
            key = (waypoint, origin, frame)
            if key in Path._PATH_CACHE:
                result = Path._PATH_CACHE[key]
            else:
                # Otherwise, construct a new SpicePath for this code, origin, and frame
                result = SpicePath(code, origin, frame, path_id=path_id)

        # The constructor above receives the integer code, so it never sees the name the
        # caller used; register that name here instead
        if isinstance(spice_path, str):
            SpicePath._register_alias(spice_path, result)

        return result

    def _get_shortcut(self, origin, frame):
        """A Path that directly transforms from the given origin and frame to this Path.

        This is an override of the default method, needed because the SPICE Toolkit can
        handle the connections between SpicePaths and the SSB very efficiently.

        Parameters:
            origin (Path): The origin Path, which must be a valid waypoint.
            frame (Frame): The Frame, which must be a valid wayframe.

        Returns:
            Path: This Path relative to `origin` and `frame`, connected through the
            nearest SpicePath ancestor of `origin`.
        """

        # Find the first SpicePath that's an ancestor of the origin
        ancestor_origin = origin
        while not isinstance(ancestor_origin, (SpicePath, SSBPath)):
            ancestor_origin = ancestor_origin._origin

        # Find the first SpiceFrame that's an ancestor of the frame
        ancestor_frame = frame
        while not isinstance(ancestor_frame, (SpiceFrame, J2000Frame)):
            ancestor_frame = ancestor_frame._reference

        # Get the SpicePath to the ancestor origin and frame
        spice_path = SpicePath.get(self, ancestor_origin, ancestor_frame)

        # Maybe we're done
        if ancestor_origin == origin and ancestor_frame == frame:
            return spice_path

        # Get the "remainder" frame from the ancestor, then link
        remainder = ancestor_origin._wrt(origin, frame, use_shortcuts=False)
        return LinkedPath(spice_path, remainder)

##########################################################################################

Path._PATH_SUBCLASSES.append(SpicePath)
Path._SpicePath = SpicePath
Frame._SpicePath = SpicePath

##########################################################################################
