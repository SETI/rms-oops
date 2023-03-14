################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

from oops.backplane import Backplane
from oops.constants import C

def distance(self, event_key, direction='dep'):
    """Distance in km between a photon's departure and its arrival.

    Input:
        event_key       key defining the surface event.
        direction       'arr' for distance traveled by the arriving photon;
                        'dep' for distance traveled by the departing photon.
    """

    if direction not in ('dep', 'arr'):
        raise ValueError('invalid photon direction: ' + repr(direction))

    event_key = self.standardize_event_key(event_key)
    key = ('distance', event_key, direction)
    if key in self.backplanes:
        return self.get_backplane(key)

    lt = self.light_time(event_key, direction)
    return self.register_backplane(key, lt * C)

#===============================================================================
def light_time(self, event_key, direction='dep'):
    """Time in seconds between a photon's departure and its arrival.

    Input:
        event_key       key defining the surface event.
        direction       'arr' for the travel time of the arriving photon;
                        'dep' for the travel time of the departing photon.
    """

    if direction not in ('dep', 'arr'):
        raise ValueError('invalid photon direction: ' + repr(direction))

    event_key = self.standardize_event_key(event_key)
    key = ('light_time', event_key, direction)
    if key in self.backplanes:
        return self.get_backplane(key)

    if direction == 'arr':
        event = self.get_surface_event(event_key, arrivals=True)
        lt = event.arr_lt
    else:
        event = self.get_surface_event(event_key)
        lt = event.dep_lt

    return self.register_backplane(key, lt.abs())

#===============================================================================
def event_time(self, event_key):
    """Absolute time in seconds TDB when the photon intercepted the surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('event_time', event_key)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key)
    return self.register_backplane(key, event.time)

#===============================================================================
def center_distance(self, event_key, direction='dep'):
    """Gridless distance traveled by a photon between paths.

    Input:
        event_key       key defining the event at the body's path.
        direction       'arr' or 'sun' to return the distance traveled by an
                                       arriving photon;
                        'dep' or 'obs' to return the distance traveled by a
                                       departing photon.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.distance(gridless_key, direction=direction)

#===============================================================================
def center_light_time(self, event_key, direction='dep'):
    """Gridless light travel time in seconds from a path.

    Input:
        event_key       key defining the event at the body's path.
        direction       'arr' or 'sun' to return the distance traveled by an
                                       arriving photon;
                        'dep' or 'obs' to return the distance traveled by a
                                       departing photon.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.light_time(gridless_key, direction=direction)

#===============================================================================
def center_time(self, event_key):
    """Gridless absolute time when the photon intercepted the path.

    Measured in seconds TDB.

    Input:
        event_key       key defining the event at the body's path.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.event_time(gridless_key)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite
import numpy as np

def distance_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:

        # Observer distance and light time
        bpt.gmtest(bp.distance(name),
                   name + ' distance to observer (km)',
                   limit=1., radius=1)
        bpt.gmtest(bp.center_distance(name),
                   name + ' center distance to observer (km)',
                   limit=1.)

        lt = bp.light_time(name)
        clt = bp.center_light_time(name)
        bpt.gmtest(lt,
                   name + ' light time to observer (s)',
                   limit=3.e-6, radius=1)
        bpt.gmtest(clt,
                   name + ' center light time to observer (km)',
                   limit=3.e-6)

        # Sun distance and light time
        bpt.gmtest(bp.distance(name, direction='arr'),
                   name + ' distance from Sun (km)',
                   limit=1., radius=1)
        bpt.gmtest(bp.center_distance(name, direction='arr'),
                    name + ' center distance from Sun (km)',
                   limit=1.)

        bpt.gmtest(bp.light_time(name, direction='arr'),
                   name + ' light time from Sun (km)',
                   limit=3.e-6, radius=1)
        bpt.gmtest(bp.center_light_time(name, direction='arr'),
                   name + ' center light time from Sun (km)',
                   limit=3.e-6, radius=1)

        # Event time
        bpt.gmtest(bp.event_time(name),
                   name + ' event time (TDB)',
                   limit=0.01, radius=1)

    for (planet, ring) in bpt.planet_ring_pairs:
        bpt.compare(bp.center_distance(planet) - bp.center_distance(ring),
                    0.,
                    planet + ' center minus ' + ring
                           + ' center to observer (km)',
                    limit=1.e-6)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.body_names + bpt.ring_names:

        km_per_los_radian = bp.distance(name) / bp.mu(name)
        (ulimit, vlimit) = km_per_los_radian.median() * pixel_duv * 1.e-4

        dist = bp.distance(name)
        ddist_duv = dist.d_dlos.chain(bp.dlos_duv)
        (ddist_du, ddist_dv) = ddist_duv.extract_denoms()

        ddist = bp_u1.distance(name) - bp_u0.distance(name)
        if not np.all(ddist.mask):
            bpt.compare((ddist.wod/bpt.duv - ddist_du).abs().median(), 0.,
                        name + ' distance d/du self-check (km/pix)',
                        limit=ulimit)

        ddist = bp_v1.distance(name) - bp_v0.distance(name)
        if not np.all(ddist.mask):
            bpt.compare((ddist.wod/bpt.duv - ddist_dv).abs().median(), 0.,
                        name + ' distance d/dv self-check (km/pix)',
                        limit=vlimit)

register_test_suite('distance', distance_test_suite)

################################################################################
