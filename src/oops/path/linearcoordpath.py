##########################################################################################
# oops/path/linearcoordpath.py
##########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path


class LinearCoordPath(Path):
    """A Path subclass defined by coordinates changing linearly on a specified
    Surface.
    """

    _WAYPOINTS = {}

    def __init__(self, surface, coords, coords_dot, epoch, *, obs=None, path_id=None):
        """Constructor for a LinearCoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            coords_dot (tuple): The time-derivatives of `coords`.
            epoch (Scalar or float): Reference time TDB for the linear motion.
            obs (Path or str, optional): The Path or the ID of the Path of the observer,
                required if `surface` is "virtual".
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            KeyError: If `obs` is an ID string that has not been registered.
            NotImplementedError: If `surface` is "virtual", meaning that its construction
                depends on the position of the observer, and `obs` is None.
            ValueError: If the shapes of `coords`, `coords_dot`, `epoch`, and `obs`
                cannot be broadcasted.
        """

        if surface.IS_VIRTUAL and obs is None:
            raise NotImplementedError('LinearCoordPath requires an observation path for '
                                      f'virtual surface class {type(surface).__name__}')

        self._surface = surface
        self._coords = tuple(Scalar.as_scalar(c).wod.as_readonly() for c in coords)
        self._coords_dot = tuple(Scalar.as_scalar(c).wod.as_readonly()
                                 for c in coords_dot)
        self._epoch = Scalar.as_scalar(epoch).wod.as_readonly()
        self._obs_path = obs and Path.as_path(obs)

        # Required attributes
        self._origin = self._surface.origin
        self._frame = self._surface.frame
        self._shape = Qube.broadcasted_shape(*self._coords, *self._coords_dot,
                                             self._epoch, self._obs_path)

        self._register(path_id)
        self.refresh()

    def _waypoint_key(self):
        return (self._surface, self._coords, self._coords_dot, self._epoch,
                self._obs_path)

    def _show(self, level, indent=0):
        name = type(self).__name__
        skip = indent + len(name) + 1
        blanks = skip * ' '

        coord_strs = [coord.mvals for coord in self._coords]
        parts = [f'{name}(surface = {self._surface}',
                 f'{blanks}coords = ({coord_strs[0]}']
        for coord_str in coord_strs[1:-1]:
            parts.append(f'{blanks}           {coord_str}')
        parts.append(f'{blanks}           {coord_strs[-1]})')

        coord_strs = [coord.mvals for coord in self._coords_dot]
        parts.append(f'{blanks}coords_dot = ({coord_strs[0]}')
        for coord_str in coord_strs[1:-1]:
            parts.append(f'{blanks}              {coord_str}')
        parts.append(f'{blanks}              {coord_strs[-1]})')

        parts.append(f'{blanks}epoch = {self._epoch},')
        if self._obs_path:
            parts.append(f'{blanks}obs = {self._obs_path.show(level-1, indent+6)}')

        return ',\n'.join(parts)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._surface, self._coords, self._coords_dot, self._epoch,
                self._obs_path, self.stripped_id)

    def __setstate__(self, state):
        (surface, coords, coords_dot, epoch, obs, path_id) = state
        self.__init__(surface, coords, coords_dot, epoch, obs=obs,
                      path_id=path_id)
        self.freeze()

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

        Raises:
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        new_coords = []
        for i in range(len(self._coords)):
            coord = self._coords[i] + self._coords_dot[i] * (time - self._epoch)
            coord.insert_deriv('t', self._coords_dot[i])
            new_coords.append(coord)

        new_coords = tuple(new_coords)
        if self._obs_path is None:
            obs = None
        else:
            obs_event = self._obs_path.event_at_time(time, quick=quick)
            obs = obs_event.wrt(self._origin, self._frame, quick=quick).pos

        pos = self._surface.vector3_from_coords(new_coords, obs=obs, derivs=True)
        return Event(time, pos, self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(LinearCoordPath)

##########################################################################################
