##########################################################################################
# oops/path/coordpath.py
##########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path


class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    _WAYPOINTS = {}

    def __init__(self, surface, coords, *, obs=None, path_id=None):
        """Constructor for a CoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            obs (Path or str, optional): The Path or ID of the observer, required if
                `surface` is "virtual".
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            KeyError: If `obs` is an ID string that has not been registered.
            NotImplementedError: If `surface` is "virtual", meaning that its construction
                depends on the position of the observer, and `obs` is None.
            ValueError: If the shapes of `coords` and `obs` cannot be broadcasted.
        """

        if surface.IS_VIRTUAL and obs is None:
            raise NotImplementedError('CoordPath requires an observation path for '
                                      f'virtual surface class {type(surface).__name__}')

        self._surface = surface
        self._coords = tuple(Scalar.as_scalar(c).as_readonly() for c in coords)
        self._obs_path = obs and Path.as_path(obs)

        # The position on a virtual surface depends on where the observer is, so it can
        # only be evaluated once the time is known; otherwise it is fixed.
        if surface.IS_VIRTUAL:
            self._pos = None
        else:
            self._pos = self._surface.vector3_from_coords(self._coords)

        self._origin = self._surface.origin
        self._frame = self._surface.frame
        self._shape = Qube.broadcasted_shape(*self._coords, self._pos, self._obs_path)

        self._register(path_id)
        self.refresh()

    def _waypoint_key(self):
        return (self._surface, self._coords, self._obs_path)

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

        if self._obs_path:
            parts.append(f'{blanks}obs = {self._obs_path.show(level-1, skip+6)}')

        return ',\n'.join(parts) + ')'

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        self.refresh()
        return (self._surface, self._coords, self._obs_path, self.stripped_id)

    def __setstate__(self, state):
        (surface, coords, obs, path_id) = state
        self.__init__(surface, coords, obs=obs, path_id=path_id)
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

        if self._pos is None:
            obs_event = self._obs_path.event_at_time(time, quick=quick)
            obs = obs_event.wrt(self._origin, self._frame, quick=quick).pos
            pos = self._surface.vector3_from_coords(self._coords, obs=obs)
        else:
            pos = self._pos

        return Event(time, pos, self._origin, self._frame)

    def _solve_photon(self, link, sign, *, derivs=False, guess=None, antimask=None,
                      quick=None, converge=None):
        """Override of the default method to avoid extra iteration."""

        return self._surface._solve_photon_by_coords(link, self._coords, sign,
                                                     derivs=derivs, guess=guess,
                                                     antimask=antimask, quick=quick,
                                                     converge=converge)

##########################################################################################

Path._PATH_SUBCLASSES.append(CoordPath)

##########################################################################################
