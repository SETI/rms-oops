################################################################################
# oops/path/linearcoordpath.py: Subclass LinearCoordPath of class Path
################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path

class LinearCoordPath(Path):
    """A path defined by coordinates changing linearly on a specified Surface.
    """

    # Note: LinearCoordPaths are not generally re-used, so their IDs are
    # expendable. Their IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, surface, coords, coords_dot, epoch, obs=None,
                       path_id=None):
        """Constructor for a LinearCoordPath.

        Input:
            surface     a surface.
            coords      a tuple of 2 or 3 Scalars defining the coordinates on
                        the surface.
            coords_dot  the time-derivative of the coords.
            epoch       the epoch at which the coords are defined, seconds TDB.
            obs         optional path of observer, needed to calculate points
                        on virtual surfaces.
            path_id     the name under which to register the new path; None to
                        leave the path unregistered.
        """

        self.surface = surface
        self.coords = [Scalar.as_scalar(c) for c in coords]
        self.coords_dot = [Scalar.as_scalar(c) for c in coords_dot]
        self.epoch = Scalar.as_scalar(epoch)
        self.obs_path = Path.as_path(obs)

        # Required attributes
        self.path_id = path_id
        self.origin  = self.surface.origin
        self.frame   = self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.surface, *self.coords,
                                              *self.coords_dot, self.epoch,
                                              self.obs_path)

        # Update waypoint and path_id; register only if necessary
        self.register()

    # Unpickled paths will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.surface, self.coords, self.coords_dot, self.epoch,
                Path.as_primary_path(self.obs_path))

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    def event_at_time(self, time, quick={}):
        """An Event corresponding to a specified time on this path.

        Input:
            time        a time Scalar at which to evaluate the path.

        Return:         an Event object containing (at least) the time, position
                        and velocity on the path.
        """

        new_coords = []
        for i in range(len(self.coords)):
            coord = self.coords[i] + self.coords_dot[i] * (time - self.epoch)
            coord.insert_deriv('t', self.coords_dot[i])
            new_coords.append(coord)

        new_coords = tuple(new_coords)

        if self.surface.IS_VIRTUAL:
            obs = self.obs_path.event_at_time(time, quick=quick).pos
        else:
            obs = None

        pos = self.surface.vector3_from_coords(new_coords, obs, derivs=True)

        return Event(time, pos, self.origin, self.frame)

################################################################################
