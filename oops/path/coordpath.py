#########################################################################################
# oops/path/coordpath.py: Subclass CoordPath of class Path
#########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.fittable   import Fittable_
from oops.path.path_ import Path


class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    _PATH_IDS = {}

    def __init__(self, surface, coords, obs=None, *, path_id=None):
        """Constructor for a CoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            obs (Path or str, optional): Path of observer, needed to calculate points on
                virtual surfaces.
            path_id (str, optional): The ID to use; None to leave the path unregistered.
        """

        if surface.IS_VIRTUAL and obs is None:
            raise NotImplementedError('CoordPath requires an observation path for '
                                      'virtual surface class ' + type(surface).__name__)

        self.surface = surface
        self.coords = tuple(Scalar(x) for x in coords)
        self.obs_path = None if obs is None else Path.as_path(obs)
        self.pos = self.surface.vector3_from_coords(self.coords)

        # Required attributes
        self.origin = self.surface.origin
        self.frame = self.origin.frame
        self.shape = Qube.broadcasted_shape(self.obs_path, *self.coords)
        self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.radius, self.lon, self.rate, self.epoch, self.origin, self.frame)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.surface, self.coords,
                None if self.obs_path is None else Path.as_primary_path(self.obs_path),
                self._state_id())

    def __setstate__(self, state):
        (surface, coords, obs, path_id) = state
        self.__init__(surface, coords, obs, path_id=path_id)
        Fittable_.freeze(self)

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

        return Event(time, self.pos, self.origin, self.frame)

    def _solve_photon(self, link, sign, derivs=False, guess=None, antimask=None, quick={},
                      converge={}):
        """Override of the default method to avoid extra iteration."""

        return self.surface._solve_photon_by_coords(link, self.coords, sign,
                                                    derivs=derivs, guess=guess,
                                                    antimask=antimask, quick=quick,
                                                    converge=converge)

#########################################################################################
