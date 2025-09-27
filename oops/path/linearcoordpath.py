##########################################################################################
# oops/path/linearcoordpath.py: Subclass LinearCoordPath of class Path
##########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.fittable   import Fittable_
from oops.path.path_ import Path


class LinearCoordPath(Path):
    """A path defined by coordinates changing linearly on a specified Surface."""

    _PATH_IDS = {}

    def __init__(self, surface, coords, coords_dot, epoch, obs=None,
                       path_id=None):
        """Constructor for a LinearCoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            coords_dot (tuple): The time-derivatives of `coords`.
            obs (Path or str, optional): Path of observer, needed to calculate points on
                virtual surfaces.
            path_id (str, optional): The ID to use; None to leave the path unregistered.
        """

        if surface.IS_VIRTUAL and obs is None:
            raise NotImplementedError('LinearCoordPath requires an observation path for '
                                      'virtual surface class ' + type(surface).__name__)

        self.surface = surface
        self.coords = [Scalar.as_scalar(c) for c in coords]
        self.coords_dot = [Scalar.as_scalar(c) for c in coords_dot]
        self.epoch = Scalar.as_scalar(epoch)
        self.obs_path = Path.as_path(obs)

        # Required attributes
        self.origin = self.surface.origin
        self.frame = self.origin.frame
        self.shape = Qube.broadcasted_shape(self.surface, *self.coords, *self.coords_dot,
                                            self.epoch, self.obs_path)
        self.path_id = self._recover_id(path_id)

        self.register()
        self._cache_id()

    ######################################################################################
    # Serialization support
    ######################################################################################

    def _path_key(self):
        return (self.surface, *self.coords, *self.coords_deriv, self.epoch, self.obs)

    def __getstate__(self):
        Fittable_.refresh(self)
        self._cache_id()
        return (self.surface, self.coords, self.coords_dot, self.epoch,
                Path.as_primary_path(self.obs_path), self._state_id())

    def __setstate__(self, state):
        self.__init__(*state[:-1], path_id=state[-1])
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

        new_coords = []
        for i in range(len(self.coords)):
            coord = self.coords[i] + self.coords_dot[i] * (time - self.epoch)
            coord.insert_deriv('t', self.coords_dot[i])
            new_coords.append(coord)

        new_coords = tuple(new_coords)
        pos = self.surface.vector3_from_coords(new_coords, derivs=True)
        return Event(time, pos, self.origin, self.frame)

##########################################################################################
