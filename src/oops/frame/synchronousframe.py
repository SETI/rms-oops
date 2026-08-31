##########################################################################################
# oops/frame/synchronousframe.py
##########################################################################################

from polymath       import Matrix3
from oops.frame     import Frame
from oops.transform import Transform


class SynchronousFrame(Frame):
    """A Frame subclass describing a body that always keeps its x-axis pointed toward a
    central planet and its y-axis in the negative direction of motion.

    Note that this Frame is tied to the orbital longitude of the body, so it will
    (incorrectly) rotate at a slightly non-uniform rate if the orbit has eccentricity.
    """

    _WAYFRAMES = {}

    def __init__(self, orbit_path, planet_path=None, *, frame_id=None):
        """Constructor for a SynchronousFrame.

        Parameters:
            orbit_path (Path or str): The Path or the ID of the Path of the body orbiting
                its planet.
            planet_path (Path or str, optional): The Path or the ID of the Path for the
                central planet. By default, this is the origin of `orbit_path`.
            frame_id (str, optional): The ID under which to register this Frame; None to
                leave this Frame unregistered. As a special case, use "+" to automatically
                generate a Frame ID by appending "_SYNCHRONOUS" to the ID of `orbit_path`
                (if it has an ID).

        Raises:
            KeyError: If `orbit_path` or `planet_path` is an ID string that has not been
                registered.
            ValueError: If `planet_path` is not shapeless.
        """

        self._orbit_path = Frame._Path.as_path(orbit_path)
        self._input_planet_path = planet_path
        self._planet_path = ((planet_path and Frame._Path.as_waypoint(planet_path))
                             or self._orbit_path._origin)
        self._orbit_wrt_planet = self._orbit_path.wrt(self._planet_path)

        if self._planet_path._shape:
            raise ValueError('SynchronousFrame requires a shapeless body path')

        self._reference = Frame.as_wayframe(self._planet_path._frame)
        self._USE_QUICKFRAMES = (self._reference._USE_QUICKFRAMES
                                 or self._planet_path._USE_QUICKPATHS
                                 or self._orbit_path._USE_QUICKPATHS)
        self._origin = self._planet_path
        self._shape = self._orbit_path._shape

        if frame_id == '+' and self._orbit_path._path_id:
            frame_id = self._orbit_path._path_id + '_SYNCHRONOUS'

        self._register(frame_id)
        self.refresh()

    def _wayframe_key(self):
        return (self._orbit_path, self._planet_path)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1

        if self._input_planet_path is None:
            return f'{name}({self._orbit_path.show(level-1, skip)})'

        return (f'{name}({self._orbit_path.show(level-1, skip)},\n'
                f'{name}({self._planet_path.show(level-1, skip)})')

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._orbit_path, self._planet_path, self.stripped_id)

    def __setstate__(self, state):
        (orbit_path, planet_path, frame_id) = state
        self.__init__(orbit_path, planet_path, frame_id=frame_id)
        self.freeze()

    ######################################################################################
    # Frame API
    ######################################################################################

    def transform_at_time(self, time, *, quick=None):
        """Transform that rotates coordinates from the reference to this frame.

        If the frame is rotating, then the coordinates being transformed must be given
        relative to the center of rotation.

        Parameters:
            time (Scalar): The time in seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters.
                Use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            Transform: Rotates vectors from the reference frame to this frame at the
            specified time.

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        event = self._orbit_wrt_planet.event_at_time(time, quick=quick)
        matrix = Matrix3.twovec(event.pos, 0, event.vel, 1)
        omega = event.pos.cross(event.vel) / event.pos.dot(event.pos)

        return Transform(matrix, omega, self, self._reference, origin=self._orbit_path)

##########################################################################################

Frame._FRAME_SUBCLASSES.append(SynchronousFrame)

##########################################################################################
