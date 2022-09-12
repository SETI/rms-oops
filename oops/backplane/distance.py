################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

import numpy as np
from polymath import Qube, Boolean, Scalar, Pair, Vector
from polymath import Vector3, Matrix3, Quaternion

from . import Backplane
from ..constants import C

#===============================================================================
def distance(self, event_key, direction='dep'):
    """Distance in km between a photon's departure and its arrival.

    Input:
        event_key       key defining the surface event.
        direction       'arr' for distance traveled by the arriving photon;
                        'dep' for distance traveled by the departing photon.
    """

    event_key = self.standardize_event_key(event_key)
    assert direction in ('dep', 'arr')

    key = ('distance', event_key, direction)
    if key not in self.backplanes:
        lt = self.light_time(event_key, direction)
        self.register_backplane(key, lt * C)

    return self.backplanes[key]

#===============================================================================
def light_time(self, event_key, direction='dep'):
    """Time in seconds between a photon's departure and its arrival.

    Input:
        event_key       key defining the surface event.
        direction       'arr' for the travel time of the arriving photon;
                        'dep' for the travel time of the departing photon.
    """

    event_key = self.standardize_event_key(event_key)
    assert direction in ('dep', 'arr')

    key = ('light_time', event_key, direction)
    if key not in self.backplanes:
        if direction == 'arr':
            event = self.get_surface_event_with_arr(event_key)
            lt = event.arr_lt
        else:
            event = self.get_surface_event(event_key)
            lt = event.dep_lt

        self.register_backplane(key, abs(lt))

    return self.backplanes[key]

#===============================================================================
def event_time(self, event_key):
    """Absolute time in seconds TDB when the photon intercepted the surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('event_time', event_key)
    if key not in self.backplanes:
        event = self.get_surface_event(event_key)
        self.register_backplane(key, event.time)

    return self.backplanes[key]

#===============================================================================
def center_distance(self, event_key, direction='dep'):
    """Distance traveled by a photon between paths.

    Input:
        event_key       key defining the event at the body's path.
        direction       'arr' or 'sun' to return the distance traveled by an
                                       arriving photon;
                        'dep' or 'obs' to return the distance traveled by a
                                       departing photon.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('center_distance', event_key, direction)
    if key not in self.backplanes:
        lt = self.center_light_time(event_key, direction)
        self.register_gridless_backplane(key, lt * C)

    return self.backplanes[key]

#===============================================================================
def center_light_time(self, event_key, direction='dep'):
    """Light travel time in seconds from a path.

    Input:
        event_key       key defining the event at the body's path.
        direction       'arr' or 'sun' to return the distance traveled by an
                                       arriving photon;
                        'dep' or 'obs' to return the distance traveled by a
                                       departing photon.
    """

    event_key = self.standardize_event_key(event_key)
    assert direction in ('dep', 'arr', 'obs', 'sun')

    key = ('center_light_time', event_key, direction)
    if key not in self.backplanes:
        if direction in ('arr', 'sun'):
            event = self.get_gridless_event_with_arr(event_key)
            lt = event.arr_lt
        else:
            event = self.get_gridless_event(event_key)
            lt = event.dep_lt

        self.register_gridless_backplane(key, abs(lt))

    return self.backplanes[key]

#===============================================================================
def center_time(self, event_key):
    """The absolute time when the photon intercepted the path.

    Measured in seconds TDB.

    Input:
        event_key       key defining the event at the body's path.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('center_time', event_key)
    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)
        self.register_gridless_backplane(key, event.time)

    return self.backplanes[key]

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################
