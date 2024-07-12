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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)
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

    event_key = Backplane.standardize_event_key(event_key)

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

    map = {'arr':'arr', 'sun':'arr', 'dep':'dep', 'obs':'dep'}
    gridless_key = Backplane.gridless_event_key(event_key)
    return self.distance(gridless_key, direction=map[direction])

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

    gridless_key = Backplane.gridless_event_key(event_key)
    return self.light_time(gridless_key, direction=direction)

#===============================================================================
def center_time(self, event_key):
    """Gridless absolute time when the photon intercepted the path.

    Measured in seconds TDB.

    Input:
        event_key       key defining the event at the body's path.
    """

    gridless_key = Backplane.gridless_event_key(event_key)
    return self.event_time(gridless_key)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
