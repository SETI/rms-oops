##########################################################################################
# oops/path/linearcoordpath.py: Subclass LinearCoordPath of class Path
##########################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path
import oops.mutable as mutable


class LinearCoordPath(Path):
    """A path defined by coordinates changing linearly on a specified Surface."""

    _WAYPOINTS = {}

    def __init__(self, surface, coords, coords_dot, epoch, *, path_id=None):
        """Constructor for a LinearCoordPath.

        Parameters:
            surface (Surface): The surface to which the coordinates refer.
            coords (tuple): 2 or 3 Scalars defining the coordinates on the surface.
            coords_dot (tuple): The time-derivatives of `coords`.
            epoch (Scalar or float): Reference time TDB for the linear motion.
            path_id (str, optional): The ID under which to register this Path; None to
                leave this Path unregistered.

        Raises:
            NotImplementedError: If this Path is "virtual", meaning that its construction
                depends on the position of the observer.
            ValueError: If the shapes of `surface`, `coords`, `coords_dot`, `epoch`, and
                `obs` cannot be broadcasted.
        """

        if surface.IS_VIRTUAL and obs is None:
            raise NotImplementedError('LinearCoordPath requires an observation path for '
                                      f'virtual surface class {surface}')

        self._surface = surface
        self._coords = tuple(Scalar.as_scalar(c).wod.as_readonly() for c in coords)
        self._coords_dot = tuple(Scalar.as_scalar(c).wod.as_readonly()
                                 for c in coords_dot)
        self._epoch = Scalar.as_scalar(epoch).wod.as_readonly()

        # Required attributes
        self._origin = self._surface._origin
        self._frame = self._origin._frame
        self._shape = Qube.broadcasted_shape(self._surface, *self._coords,
                                             *self._coords_dot, self._epoch)

        self._register(path_id)
        mutable.refresh(self)

    def _waypoint_key(self):
        return (self._surface, self._coords, self._coords_dot, self._epoch)

    ######################################################################################
    # Serialization support
    ######################################################################################

    def __getstate__(self):
        mutable.refresh(self)
        return (self._surface, self._coords, self._coords_dot, self._epoch,
                self.stripped_id)

    def __setstate__(self, state):
        (surface, coords, coords_dot, epoch, path_id) = state
        self.__init__(surface, coords, coords_dot, epoch, path_id=path_id)
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

        new_coords = []
        for i in range(len(self._coords)):
            coord = self._coords[i] + self._coords_dot[i] * (time - self._epoch)
            coord.insert_deriv('t', self._coords_dot[i])
            new_coords.append(coord)

        new_coords = tuple(new_coords)
        pos = self._surface.vector3_from_coords(new_coords, derivs=True)
        return Event(time, pos, self._origin, self._frame)

##########################################################################################

Path._PATH_SUBCLASSES.append(LinearCoordPath)

##########################################################################################
