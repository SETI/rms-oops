#########################################################################################
# oops/path/pathshift.py: Subclass PathShift of class Path
#########################################################################################

from polymath        import Scalar
from oops.fittable   import Fittable
from oops.path.path_ import Path


class PathShift(Path, Fittable):
    """A path defined by a time-shift along another path.

    PLACEHOLDER CODE. "CONCEPTUALLY" CORRECT BUT NOT YET TESTED.
    """

    _PATH_IDS = {}

    def __init__(self, dt, /, path, *, path_id=None):
        """Constructor for a PathShift.

        Parameters:
            dt (float): The initial time shift in seconds.
            path (Path or str): The Path or ID to which the time shift applies.
            path_id (str, optional): The new path ID to use; None to leave this path
                unregistered.
        """

        self.dt = dt
        self.path = path

        # Required attributes
        self.origin = self.path.origin
        self.frame = self.path.frame
        self.shape = self.path.shape
        self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Fittable interface
    ######################################################################################

    def _set_params(self, params):
        self.dt = params[0]

    @property
    def _params(self):
        return (self.dt,)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.dt, self.path)

    def __getstate__(self):
        self.refresh(self)
        self._cache_id()
        return (self.dt, Path.as_primary_path(self.path), self._state_id())

    def __setstate__(self, state):
        (dt, path, path_id) = state
        self.__init__(dt, path, path_id=path_id)
        self.freeze()

    ######################################################################################
    # Path API
    ######################################################################################

    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified time on this path.

        Parameters:
            time (Scalar, array-like, or float): Time at which to evaluate the path, in
                seconds TDB.
            quick (dict or bool, optional): A dictionary of parameter values to use as
                overrides to the configured default QuickPath and QuickFrame parameters;
                use False to disable the use of QuickPaths and QuickFrames.

        Returns:
            (Event): Event object containing (at least) the time, position, and velocity
                on the path.
        """

        time = Scalar.as_scalar(time)
        return self.path.event_at_time(time + self.dt, quick=quick)

#########################################################################################
