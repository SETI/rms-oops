################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

from polymath import Pair, Vector3

from oops.backplane import Backplane
from oops.surface   import Surface

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




################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.meshgrid     import Meshgrid
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
from oops.constants    import DPR
from oops.backplane.unittester_support    import show_info


#===========================================================================
def exercise_surface(bp, obs, printing, saving, dir, 
                     planet=None, moon=None, ring=None, 
                     undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for resolution.py"""
    
    if planet != None:
        test = bp.resolution(planet, 'u')
        show_info('planet resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(planet, 'v')
        show_info('planet resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(planet, 'u')
        show_info('planet center resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(planet, 'v')
        show_info('planet center resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.finest_resolution(planet)
        show_info('planet finest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.coarsest_resolution(planet)
        show_info('planet coarsest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.resolution(moon, 'u')
        show_info('moon resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(moon, 'v')
        show_info('moon resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(moon, 'u')
        show_info('moon center resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(moon, 'v')
        show_info('moon center resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.finest_resolution(moon)
        show_info('moon finest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.coarsest_resolution(moon)
        show_info('moon coarsest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

    if ring != None:
        test = bp.resolution(ring, 'u')
        show_info('Ring resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(ring, 'v')
        show_info('Ring resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(ring, 'u')
        show_info('Ring center resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(ring, 'v')
        show_info('Ring center resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.finest_resolution(ring)
        show_info('Ring finest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.coarsest_resolution(ring)
        show_info('Ring coarsest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)




#===========================================================================
def exercise_ansa(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for resolution.py"""
    
    if planet != None:
        test = bp.resolution(planet+':ansa', 'u')
        show_info('Ansa resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(planet+':ansa', 'v')
        show_info('Ansa resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(planet+':ansa', 'u')
        show_info('Ansa center resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_resolution(planet+':ansa', 'v')
        show_info('Ansa center resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.finest_resolution(planet+':ansa')
        show_info('Ansa finest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.coarsest_resolution(planet+':ansa')
        show_info('Ansa coarsest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)




#===========================================================================
def exercise_limb(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for resolution.py"""
    
    if planet != None:
        test = bp.resolution(planet+':limb', 'u')
        show_info('Limb resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(planet+':limb', 'v')
        show_info('Limb resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(planet+':limb', 'u')
        show_info('Limb resolution along u axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.resolution(planet+':limb', 'v')
        show_info('Limb resolution along v axis (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.finest_resolution(planet+':limb')
        show_info('Limb finest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.coarsest_resolution(planet+':limb')
        show_info('Limb coarsest resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)





#*******************************************************************************
class Test_Resolution(unittest.TestCase):


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
