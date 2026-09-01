##########################################################################################
# oops/backplane/resolution.py
##########################################################################################

from oops.backplane import Backplane
from oops.surface   import Surface


def resolution(self, event_key, axis='u'):
    """Projected resolution in km/pixel at the surface intercept.

    Defined perpendicular to the line of sight.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
        axis (str, optional): 'u' for resolution along the horizontal axis of the
            observation; 'v' for resolution along the vertical axis of the observation.
    """

    if axis not in ('u', 'v'):
        raise ValueError('invalid axis: ' + repr(axis))

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)
    key = ('resolution', event_key, axis)
    if key not in self.backplanes:
        distance = self.distance(event_key)
        (dlos_du, dlos_dv) = self.dlos_duv.extract_denoms()
        self.register_backplane(key[:-1] + ('u',), distance * dlos_du.norm())
        self.register_backplane(key[:-1] + ('v',), distance * dlos_dv.norm())

    return self.get_backplane(key)


def center_resolution(self, event_key, axis='u'):
    """Gridless, directionless projected spatial resolution in km/pixel.

    Measured at the central path of a body, based on range alone.

    Parameters:
        event_key (str or tuple): Key defining the event at the body's path.
        axis (str, optional): 'u' for resolution along the horizontal axis of the
            observation; 'v' for resolution along the vertical axis of the observation.
    """

    if axis not in ('u', 'v'):
        raise ValueError('invalid axis: ' + repr(axis))

    self.refresh()
    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('center_resolution', gridless_key, axis)
    if key not in self.backplanes:
        distance = self.center_distance(gridless_key)
        (dlos_du, dlos_dv) = self.center_dlos_duv.extract_denoms()
        self.register_backplane(key[:-1] + ('u',), distance * dlos_du.norm())
        self.register_backplane(key[:-1] + ('v',), distance * dlos_dv.norm())

    return self.get_backplane(key)


def finest_resolution(self, event_key):
    """Projected resolution in km/pixel for the optimal direction.

    Determined at the intercept point on the surface.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)
    key = ('finest_resolution', event_key)
    if key not in self.backplanes:
        self._fill_surface_resolution(event_key)

    return self.get_backplane(key)


def coarsest_resolution(self, event_key):
    """Projected spatial resolution in km/pixel in the worst direction at the intercept
    point.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
    """

    self.refresh()
    event_key = Backplane.standardize_event_key(event_key)
    key = ('coarsest_resolution', event_key)
    if key not in self.backplanes:
        self._fill_surface_resolution(event_key)

    return self.get_backplane(key)


def _fill_surface_resolution(self, event_key):
    """Internal method to fill in the surface resolution backplanes.

    Registers the finest and coarsest resolution backplanes, in km per pixel, for the
    surface intercept points.

    Parameters:
        event_key (str or tuple): Key defining the surface event.
    """

    event_key = Backplane.standardize_event_key(event_key)
    event = self.get_surface_event(event_key, derivs=True)

    dpos_duv1 = event.pos.d_dlos.chain(self.dlos_duv1)
    (minres, maxres) = Surface.resolution(dpos_duv1)
    self.register_backplane(('finest_resolution',   event_key), minres)
    self.register_backplane(('coarsest_resolution', event_key), maxres)

##########################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

##########################################################################################
