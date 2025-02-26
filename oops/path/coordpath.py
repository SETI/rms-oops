################################################################################
# oops/path/coordpath.py: Subclass CoordPath of class Path
################################################################################

from polymath        import Qube, Scalar
from oops.event      import Event
from oops.path.path_ import Path

class CoordPath(Path):
    """A path defined by fixed coordinates on a specified Surface."""

    # Note: CoordPaths are not generally re-used, so their IDs are expendable.
    # Their IDs are not preserved during pickling.

    #===========================================================================
    def __init__(self, surface, coords, obs=None, path_id=None):
        """Constructor for a CoordPath.

        Input:
            surface     a surface.
            coords      a tuple of 2 or 3 Scalars defining the coordinates on
                        the surface.
            obs         optional path of observer, needed to calculate points
                        on virtual surfaces.
            path_id     the name under which to register the new path; None to
                        leave the path unregistered.
        """

        if surface.IS_VIRTUAL:
            raise NotImplementedError('CoordPath cannot be defined for virtual '
                                      'surface class '
                                      + type(surface).__name__)

        self.surface = surface
        self.coords = tuple(Scalar(x) for x in coords)
        self.obs_path = None if obs is None else Path.as_path(obs)
        self.pos = self.surface.vector3_from_coords(self.coords)

        # Required attributes
        self.path_id = path_id
        self.origin  = self.surface.origin
        self.frame   = self.origin.frame
        self.keys    = set()
        self.shape   = Qube.broadcasted_shape(self.obs_path, *self.coords)

        # Update waypoint and path_id; register only if necessary
        self.register()

    # Unpickled paths will always have temporary IDs to avoid conflicts
    def __getstate__(self):
        return (self.surface, self.coords,
                None if self.obs_path is None
                                      else Path.as_primary_path(self.obs_path))

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

        return Event(time, self.pos, self.origin, self.frame)

    #===========================================================================
    def _solve_photon(self, link, sign, derivs=False, guess=None, antimask=None,
                            quick={}, converge={}):
        """Override of the default method to avoid extra iteration."""

        return self.surface._solve_photon_by_coords(link, self.coords, sign,
                                                    derivs=derivs, guess=guess,
                                                    antimask=antimask,
                                                    quick=quick,
                                                    converge=converge)

################################################################################
