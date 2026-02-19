#########################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
#########################################################################################

from polymath        import Scalar
from oops.fittable   import Fittable
from oops.path.path_ import Path
import oops.mutable as mutable


class PathShift(Path, Fittable):
    """A path defined by a time-shift along another path.

    PLACEHOLDER CODE. "CONCEPTUALLY" CORRECT BUT NOT YET TESTED.
    """

    _WAYPOINTS = {}

    def __init__(self, arg, /, path, *, path_id=None, freeze=False):
        """Constructor for a PathShift.

        Parameters:
            arg (float, PathShift, FrameShift, or TimeShift): The initial time shift in
                seconds. Alternatively, if another time-shifted object is given, this
                object's time shift will always match that of the argument.
            path (Path or str): The Path or ID to which the time shift applies.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered. As a special case, use "+" to automatically
                generate a Path ID by appending "_SHIFT" to the ID of `path` (if it has
                an ID).
            freeze (bool, optional): True to return a frozen object; False to leave it
                unfrozen.
        """

        # Linking to a frozen object yields a frozen object
        if isinstance(arg, str):
            arg = Path.as_path(arg)
        if hasattr(arg, 'dt') and mutable.is_frozen(arg):
            freeze = True
            arg = arg.dt

        if hasattr(arg, 'dt'):
            self._link = arg
        else:
            self._dt = arg
            self._link = None

        self._path = Path.as_path(path)
        self._origin = self._path._origin
        self._frame = self._path._frame
        self._shape = self._path._shape

        if path_id == '+' and self._path._path_id:
            path_id = self._path._path_id + '_SHIFT'

        self._register(path_id)
        self.refresh()
        if freeze:
            self.freeze()

    @property
    def dt(self):
        return self._dt

    def _source(self):
        """The original source of the time shift if this object is linked to another;
        otherwise, self.
        """
        return self._link and self._link._source() or self

    def _waypoint_key(self):
        if self.is_frozen:
            return (self._dt, self._path)
        # Use id(self) to ensure that an unlinked PathShift has a unique key
        return (self._link or id(self), self._path)

    ######################################################################################
    # Fittable interface
    ######################################################################################

    nparams = 1

    def _set_params(self, params):
        """Redefine the time offset of this PathShift object.

        If this object is linked to another, the time offset of the linked object is also
        redefined.
        """

        if self._link:
            self._link.set_params(params)
            self._dt = self._link.dt
        else:
            self._dt = params[0]

    @property
    def params(self):
        return (self._dt,)

    def _refresh(self):
        if self.link:
            self._dt = self._link.dt

    def _freeze(self):
        if self._link:
            self._dt = self._link._dt
            self._link = None
        self._reregister()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._dt, self._path, self.stripped_id)

    def __setstate__(self, state):
        (dt, path, path_id) = state
        self.__init__(dt, path, path_id=path_id)
        self.freeze()

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, *, quick=None):
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
            ValueError: If the shape of `time` cannot be broadcasted to the shape of this
                Path.
        """

        time = Scalar.as_scalar(time)
        return self._path.event_at_time(time + self._dt, quick=quick)

##########################################################################################

Path._PATH_SUBCLASSES.append(PathShift)

#########################################################################################
