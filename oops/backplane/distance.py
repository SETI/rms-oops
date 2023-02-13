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

register_test_suite('distance', distance_test_suite)

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.body                         import Body
from oops.backplane.unittester_support import show_info

#===============================================================================
def exercise_observer(bp,
                      planet=None, moon=None, ring=None,
                      undersample=16, use_inventory=False, inventory_border=2,
                      **options):
    """generic unit tests for distance.py"""

    if planet is not None:
        test = bp.distance(planet)
        show_info(bp, 'Distance observer to planet (km)', test, **options)
        test = bp.distance(planet, direction='dep')
        show_info(bp, 'Distance observer to planet via dep (km)', test, **options)
        test = bp.center_distance(planet)
        show_info(bp, 'Distance observer to planet center (km)', test, **options)
        test = bp.distance(planet+':limb')
        show_info(bp, 'Distance observer to planet limb (km)', test, **options)

        if Body.lookup(planet).ring_body is not None:
            test = bp.distance(planet+':ring')
            show_info(bp, 'Distance observer to planet:ring (km)', test, **options)
            test = bp.distance(planet+':ansa')
            show_info(bp, 'Distance observer to planet:ansa (km)', test, **options)

    if ring is not None:
        test = bp.distance(ring)
        show_info(bp, 'Distance observer to rings (km)', test, **options)
        test = bp.center_distance(ring)
        show_info(bp, 'Distance observer to ring center (km)', test, **options)

    if moon is not None:
        test = bp.distance(moon)
        show_info(bp, 'Distance observer to moon (km)', test, **options)
        test = bp.center_distance(moon)
        show_info(bp, 'Distance observer to moon center (km)', test, **options)

#===============================================================================
def exercise_sun(bp,
                 planet=None, moon=None, ring=None,
                 undersample=16, use_inventory=False, inventory_border=2,
                 **options):
    """generic unit tests for Sun distance.py"""

    if planet is not None:
        test = bp.distance(planet, direction='arr')
        show_info(bp, 'Distance Sun to planet, arrival (km)', test, **options)
        test = bp.center_distance(planet, direction='arr')
        show_info(bp, 'Distance Sun to planet center, arrival (km)', test, **options)

        if Body.lookup(planet).ring_body is not None:
            test = bp.distance(planet+':ring', direction='arr')
            show_info(bp, 'Distance Sun to planet:ring (km)', test, **options)
            test = bp.distance(planet+':ansa', direction='arr')
            show_info(bp, 'Distance Sun to planet:ansa (km)', test, **options)

        test = bp.distance(planet+':limb', direction='arr')
        show_info(bp, 'Distance Sun to limb (km)', test, **options)

    if ring is not None:
        test = bp.distance(ring, direction='arr')
        show_info(bp, 'Distance Sun to rings, arrival (km)', test, **options)
        test = bp.center_distance(ring, direction='arr')
        show_info(bp, 'Distance Sun to ring center, arrival (km)', test, **options)

#===============================================================================
def exercise_observer_light_time(bp,
                                 planet=None, moon=None, ring=None,
                                 undersample=16, use_inventory=False,
                                 inventory_border=2,
                                 **options):
    """generic unit tests for light_time.py"""

    if planet is not None:
        test = bp.light_time(planet)
        show_info(bp, 'Light-time observer to planet (sec)', test, **options)
        test = bp.light_time(planet, direction='dep')
        show_info(bp, 'Light-time observer to planet via dep (sec)', test, **options)
        test = bp.light_time(planet+':limb')
        show_info(bp, 'Light-time observer to limb (sec)', test, **options)

        if Body.lookup(planet).ring_body:
            test = bp.light_time(planet+':ring')
            show_info(bp, 'Light-time observer to planet:ring (sec)', test, **options)
            test = bp.light_time(planet+':ansa')
            show_info(bp, 'Light-time observer to planet:ansa (sec)', test, **options)

        test = bp.center_light_time(planet)
        show_info(bp, 'Light-time observer to planet center (sec)', test, **options)

    if ring is not None:
        test = bp.light_time(ring)
        show_info(bp, 'Light-time observer to rings (sec)', test, **options)
        test = bp.center_light_time(ring)
        show_info(bp, 'Light-time observer to ring center (sec)', test, **options)

    if moon is not None:
        test = bp.light_time(moon)
        show_info(bp, 'Light-time observer to moon (sec)', test, **options)
        test = bp.center_light_time(moon)
        show_info(bp, 'Light-time observer to moon center (sec)', test, **options)

#===========================================================================
def exercise_sun_light_time(bp,
                            planet=None, moon=None, ring=None,
                            undersample=16, use_inventory=False,
                            inventory_border=2,
                            **options):
    """generic unit tests for Sun light_time.py"""

    if planet is not None:
        test = bp.light_time(planet)
        show_info(bp, 'Light-time Sun to planet (sec)', test, **options)
        test = bp.light_time(planet, direction='dep')
        show_info(bp, 'Light-time Sun to planet via dep (sec)', test, **options)
        test = bp.light_time(planet+':limb')
        show_info(bp, 'Light-time Sun to limb (sec)', test, **options)

        if Body.lookup(planet).ring_body:
            test = bp.light_time(planet+':ring')
            show_info(bp, 'Light-time Sun to planet:ring (sec)', test, **options)
            test = bp.light_time(planet+':ansa')
            show_info(bp, 'Light-time Sun to planet:ansa (sec)', test, **options)

        test = bp.center_light_time(planet)
        show_info(bp, 'Light-time Sun to planet center (sec)', test, **options)

    if ring is not None:
        test = bp.light_time(ring)
        show_info(bp, 'Light-time Sun to rings (sec)', test, **options)
        test = bp.center_light_time(ring)
        show_info(bp, 'Light-time Sun to ring center (sec)', test, **options)

    if moon is not None:
        test = bp.light_time(moon)
        show_info(bp, 'Light-time Sun to moon (sec)', test, **options)
        test = bp.center_light_time(moon)
        show_info(bp, 'Light-time Sun to moon center (sec)', test, **options)

#===============================================================================
def exercise_event_time(bp,
                        planet=None, moon=None, ring=None,
                        undersample=16, use_inventory=False,
                        inventory_border=2,
                        **options):
    """generic unit tests for event_time.py"""

    test = bp.event_time(())
    show_info(bp, 'Event time at Cassini (sec, TDB)', test, **options)
    test = bp.center_time(())
    show_info(bp, 'Event time at Cassini center (sec, TDB)', test, **options)

    if planet is not None:
        test = bp.event_time(planet)
        show_info(bp, 'Event time at planet (sec, TDB)', test, **options)
        test = bp.center_time(planet)
        show_info(bp, 'Event time at planet (sec, TDB)', test, **options)
        if Body.lookup(planet).ring_body is not None:
            test = bp.center_time(planet+':ring')
            show_info(bp, 'Event time at planet:ring (sec, TDB)', test, **options)
            test = bp.center_time(planet+':ansa')
            show_info(bp, 'Event time at planet:ansa (sec, TDB)', test, **options)

    if ring is not None:
        test = bp.event_time(ring)
        show_info(bp, 'Event time at rings (sec, TDB)', test, **options)
        test = bp.center_time(ring)
        show_info(bp, ' Event time at ring center (sec, TDB)', test, **options)

    if moon is not None:
        test = bp.event_time(moon)
        show_info(bp, 'Event time at moon (sec, TDB)', test, **options)
        test = bp.event_time(moon)
        show_info(bp, 'Event time at moon (sec, TDB)', test, **options)
        test = bp.event_time(moon)
        show_info(bp, 'Event time at moon center (sec, TDB)', test, **options)


#*******************************************************************************
class Test_Distance(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        from oops.backplane.unittester_support import Backplane_Settings
        if Backplane_Settings.EXERCISES_ONLY:
            self.skipTest("")
        pass


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
