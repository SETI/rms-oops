################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from . import Backplane
from ..surface import Surface

#===============================================================================
def resolution(self, event_key, axis='u'):
    """Projected resolution in km/pixel at the surface intercept.

    Defined perpendicular to the line of sight.

    Input:
        event_key       key defining the surface event.
        axis            'u' for resolution along the horizontal axis of the
                            observation;
                        'v' for resolution along the vertical axis of the
                            observation.
    """

    event_key = self.standardize_event_key(event_key)
    assert axis in ('u','v')

    key = ('resolution', event_key, axis)
    if key not in self.backplanes:
        distance = self.distance(event_key)

        res = self.dlos_duv.swap_items(Pair)
        (u_resolution, v_resolution) = res.to_scalars()
        u_resolution = distance * u_resolution.join_items(Vector3).norm()
        v_resolution = distance * v_resolution.join_items(Vector3).norm()

        self.register_backplane(key[:-1] + ('u',), u_resolution)
        self.register_backplane(key[:-1] + ('v',), v_resolution)

    return self.backplanes[key]

#===============================================================================
def center_resolution(self, event_key, axis='u'):
    """Directionless projected spatial resolution in km/pixel.

    Measured at the central path of a body, based on range alone.

    Input:
        event_key       key defining the event at the body's path.
        axis            'u' for resolution along the horizontal axis of the
                            observation;
                        'v' for resolution along the vertical axis of the
                            observation.
    """

    event_key = self.standardize_event_key(event_key)
    assert axis in ('u','v')

    key = ('center_resolution', event_key, axis)
    if key not in self.backplanes:
        distance = self.center_distance(event_key)

        res = self.obs.fov.center_dlos_duv.swap_items(Pair)
        (u_resolution, v_resolution) = res.to_scalars()
        u_resolution = distance * u_resolution.join_items(Vector3).norm()
        v_resolution = distance * v_resolution.join_items(Vector3).norm()

        self.register_gridless_backplane(key[:-1] + ('u',), u_resolution)
        self.register_gridless_backplane(key[:-1] + ('v',), v_resolution)

    return self.backplanes[key]

#===============================================================================
def finest_resolution(self, event_key):
    """Projected resolution in km/pixel for the optimal direction

    Determined a the intercept point on the surface.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('finest_resolution', event_key)
    if key not in self.backplanes:
        self._fill_surface_resolution(event_key)

    return self.backplanes[key]

#===============================================================================
def coarsest_resolution(self, event_key):
    """Projected spatial resolution in km/pixel in the worst direction at the
    intercept point.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('coarsest_resolution', event_key)
    if key not in self.backplanes:
        self._fill_surface_resolution(event_key)

    return self.backplanes[key]

#===============================================================================
def _fill_surface_resolution(self, event_key):
    """Internal method to fill in the surface resolution backplanes."""

    event_key = self.standardize_event_key(event_key)
    event = self.get_surface_event_w_derivs(event_key)

    dpos_duv = event.state.d_dlos.chain(self.dlos_duv)
    (minres, maxres) = Surface.resolution(dpos_duv)

    self.register_backplane(('finest_resolution', event_key), minres)
    self.register_backplane(('coarsest_resolution', event_key), maxres)

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################
