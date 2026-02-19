#########################################################################################
# oops/path/coordpath.py: Subclass CoordPath of class Path
#########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path
import oops.mutable as mutable


class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    _WAYPOINTS = {}

    def __init__(self, surface, coords, *, path_id=None):
        """Constructor for a CoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            NotImplementedError: If `surface` is "virtual", meaning that its construction
                depends on the position of the observer.
            ValueError: If the shapes of `surface`, `coords`, and `pos` cannot be
                broadcasted.
        """

        if surface._IS_VIRTUAL and obs is None:
            raise NotImplementedError('CoordPath requires an observation path for '
                                      f'virtual surface class {type(surface).__name__}')

        self._surface = surface
        self._coords = tuple(Scalar(x).as_readonly() for x in coords)
        self._pos = self.surface.vector3_from_coords(self._coords)

        self._origin = self._surface._origin
        self._frame = self._origin._frame
        self._shape = Qube.broadcasted_shape(self._surface.shape, self._coords[0],
                                             self._pos)

        self._register(path_id)
        mutable.refresh(self)

    def _waypoint_key(self):
        return (self._surface, self._coords)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._surface, self._coords, self.stripped_id)

    def __setstate__(self, state):
        (surface, coords, path_id) = state
        self.__init__(surface, coords, path_id=path_id)
        mutable.freeze(self)

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
            ValueError: If the shapes of `time` and this object cannot be broadcasted.
        """

        return Event(time, self._pos, self._origin, self._frame)

    def _solve_photon(self, link, sign, derivs=False, guess=None, antimask=None,
                      quick=None, converge=None):
        """Override of the default method to avoid extra iteration."""

        return self.surface._solve_photon_by_coords(link, self._coords, sign,
                                                    derivs=derivs, guess=guess,
                                                    antimask=antimask, quick=quick,
                                                    converge=converge)

##########################################################################################

Path._PATH_SUBCLASSES.append(CoordPath)

#########################################################################################
