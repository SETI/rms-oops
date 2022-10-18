################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

from oops.backplane import Backplane
from oops.constants import C

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




################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.meshgrid                      import Meshgrid
from oops.unittester_support import     TESTDATA_PARENT_DIRECTORY
from oops.constants                     import DPR
from oops.backplane.unittester_support  import show_info


#===========================================================================
def exercise_observer(bp, obs, printing, saving, dir, 
                      planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for distance.py"""
    
    if planet != None:
        test = bp.distance(planet)
        show_info('Distance observer to planet (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(planet, direction='dep')
        show_info('Distance observer to planet via dep (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(planet)
        show_info('Distance observer to planet center (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(planet+':limb')
        show_info('Distance observer to planet limb (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(planet+':ansa')
        show_info('Distance observer to ansa (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.distance(ring)
        show_info('Distance observer to rings (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(ring)
        show_info('Distance observer to ring center (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.distance(moon)
        show_info('Distance observer to moon (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(moon)
        show_info('Distance observer to moon center (km)', test,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_sun(bp, obs, printing, saving, dir, 
                      planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for distance.py"""
    
    if planet != None:
        test = bp.distance(planet, direction='arr')
        show_info('Distance Sun to planet, arrival (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(('sun', planet), direction='dep')
        show_info('Distance Sun to planet, departure (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(planet, direction='arr')
        show_info('Distance Sun to planet center, arrival (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(('sun', planet), direction='dep')
        show_info('Distance Sun to planet center, departure (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(planet+':ansa', direction='arr')
        show_info('Distance Sun to ansa (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(planet+':limb', direction='arr')
        show_info('Distance Sun to limb (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.distance(ring, direction='arr')
        show_info('Distance Sun to rings, arrival (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.distance(('sun', ring), direction='dep')
        show_info('Distance Sun to rings, departure (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(ring, direction='arr')
        show_info('Distance Sun to ring center, arrival (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_distance(('sun', ring), direction='dep')
        show_info('Distance Sun to ring center, departure (km)', test,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_observer_light_time(bp, obs, printing, saving, dir, 
                      planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for distance.py"""
    
    if planet != None:
        test = bp.light_time(planet)
        show_info('Light-time observer to planet (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet, direction='dep')
        show_info('Light-time observer to planet via dep (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet+':limb')
        show_info('Light-time observer to limb (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet+':ansa')
        show_info('Light-time observer to ansa (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(planet)
        show_info('Light-time observer to planet center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.light_time(ring)
        show_info('Light-time observer to rings (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(ring)
        show_info('Light-time observer to ring center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.light_time(moon)
        show_info('Light-time observer to moon (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(moon)
        show_info('Light-time observer to moon center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_sun_light_time(bp, obs, printing, saving, dir, 
                      planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for distance.py"""
    
    if planet != None:
        test = bp.light_time(planet)
        show_info('Light-time observer to planet (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet, direction='dep')
        show_info('Light-time observer to planet via dep (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet+':limb')
        show_info('Light-time observer to limb (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.light_time(planet+':ansa')
        show_info('Light-time observer to ansa (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(planet)
        show_info('Light-time observer to planet center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.light_time(ring)
        show_info('Light-time observer to rings (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(ring)
        show_info('Light-time observer to ring center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.light_time(moon)
        show_info('Light-time observer to moon (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_light_time(moon)
        show_info('Light-time observer to moon center (sec)', test,   
                                    printing=printing, saving=saving, dir=dir)




#===========================================================================
def exercise_event_time(bp, obs, printing, saving, dir, 
                      planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for distance.py"""
    
    test = bp.event_time(())
    show_info('Event time at Cassini (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

    test = bp.center_time(())
    show_info('Event time at Cassini center (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if planet != None:
        test = bp.event_time(planet)
        show_info('Event time at planet (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_time(planet)
        show_info('Event time at planet (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.event_time(ring)
        show_info('Event time at rings (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_time(ring)
        show_info(' Event time at ring center (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.event_time(moon)
        show_info('Event time at moon (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.event_time(moon)
        show_info('Event time at moon (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.event_time(moon)
        show_info('Event time at moon center (sec, TDB)', test,   
                                    printing=printing, saving=saving, dir=dir)





#*******************************************************************************
class Test_Distance(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        from oops.backplane.unittester_support import Backplane_Settings
        if Backplane_Settings.EXERCISES_ONLY: 
            return
        pass


########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
