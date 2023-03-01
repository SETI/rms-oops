################################################################################
# oops/backplanes/distance.py: Distance-related backplanes
################################################################################

from polymath       import Pair, Vector3
from oops.backplane import Backplane
from oops.surface   import Surface

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

    if axis not in ('u', 'v'):
        raise ValueError('invalid axis: ' + repr(axis))

    event_key = self.standardize_event_key(event_key)
    key = ('resolution', event_key, axis)
    if key not in self.backplanes:
        distance = self.distance(event_key)

        dlos_du = Vector3(self.dlos_duv.vals[...,0], self.dlos_duv.mask)
        dlos_dv = Vector3(self.dlos_duv.vals[...,1], self.dlos_duv.mask)

        self.register_backplane(key[:-1] + ('u',), distance * dlos_du.norm())
        self.register_backplane(key[:-1] + ('v',), distance * dlos_dv.norm())

    return self.get_backplane(key)

#===============================================================================
def center_resolution(self, event_key, axis='u'):
    """Gridless, directionless projected spatial resolution in km/pixel.

    Measured at the central path of a body, based on range alone.

    Input:
        event_key       key defining the event at the body's path.
        axis            'u' for resolution along the horizontal axis of the
                            observation;
                        'v' for resolution along the vertical axis of the
                            observation.
    """

    if axis not in ('u', 'v'):
        raise ValueError('invalid axis: ' + repr(axis))

    gridless_key = self.gridless_event_key(event_key)
    key = ('center_resolution', gridless_key, axis)
    if key not in self.backplanes:
        distance = self.center_distance(gridless_key)

        mask = self.center_dlos_duv.mask
        dlos_du = Vector3(self.center_dlos_duv.vals[...,0], mask)
        dlos_dv = Vector3(self.center_dlos_duv.vals[...,1], mask)

        self.register_backplane(key[:-1] + ('u',), distance * dlos_du.norm())
        self.register_backplane(key[:-1] + ('v',), distance * dlos_dv.norm())

    return self.get_backplane(key)

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

    return self.get_backplane(key)

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

    return self.get_backplane(key)

#===============================================================================
def _fill_surface_resolution(self, event_key):
    """Internal method to fill in the surface resolution backplanes."""

    event_key = self.standardize_event_key(event_key)
    event = self.get_surface_event(event_key, derivs=True)

    dpos_duv = event.state.d_dlos.chain(self.dlos_duv)
    (minres, maxres) = Surface.resolution(dpos_duv)

    self.register_backplane(('finest_resolution',   event_key), minres)
    self.register_backplane(('coarsest_resolution', event_key), maxres)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite

def resolution_test_suite(bpt):

    bp = bpt.backplane
    for name in (bpt.body_names + bpt.limb_names +
                 bpt.ring_names + bpt.ansa_names):

        bpt.gmtest(bp.resolution(name, 'u'),
                   name + ' resolution along u axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.resolution(name, 'v'),
                   name + ' resolution along v axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.center_resolution(name, 'u'),
                   name + ' center resolution along u axis (km)',
                   limit=0.01, radius=1.5)
        bpt.gmtest(bp.center_resolution(name, 'v'),
                   name + ' center resolution along v axis (km)',
                   limit=0.01, radius=1.5)

        # Because finest/coarsest resolution values diverge for emission angles
        # near 90, we need to apply an extra mask
        finest = bp.finest_resolution(name)
        coarsest = bp.coarsest_resolution(name)
        mu = bp.emission_angle(name).cos().abs()
        mask = mu.tvl_lt(0.1).as_mask_where_nonzero_or_masked()

        bpt.gmtest(bp.finest_resolution(name),
                   name + ' finest resolution (km)',
                   limit=0.01, radius=1.5, mask=mask)
        bpt.gmtest(bp.coarsest_resolution(name),
                   name + ' coarsest resolution (km)',
                   limit=0.1, radius=1.5, mask=mask)

register_test_suite('resolution', resolution_test_suite)

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.body import Body
from oops.backplane.unittester_support import show_info

#===========================================================================
def exercise_surface(bp,
                     planet=None, moon=None, ring=None,
                     undersample=16, use_inventory=False, inventory_border=2,
                     **options):
    """generic unit tests for resolution.py"""

    if planet is not None:
        test = bp.resolution(planet, 'u')
        show_info(bp, 'planet resolution along u axis (km)', test, **options)
        test = bp.resolution(planet, 'v')
        show_info(bp, 'planet resolution along v axis (km)', test, **options)
        test = bp.center_resolution(planet, 'u')
        show_info(bp, 'planet center resolution along u axis (km)', test, **options)
        test = bp.center_resolution(planet, 'v')
        show_info(bp, 'planet center resolution along v axis (km)', test, **options)
        test = bp.finest_resolution(planet)
        show_info(bp, 'planet finest resolution (km)', test, **options)
        test = bp.coarsest_resolution(planet)
        show_info(bp, 'planet coarsest resolution (km)', test, **options)

    if moon is not None:
        test = bp.resolution(moon, 'u')
        show_info(bp, 'moon resolution along u axis (km)', test, **options)
        test = bp.resolution(moon, 'v')
        show_info(bp, 'moon resolution along v axis (km)', test, **options)
        test = bp.center_resolution(moon, 'u')
        show_info(bp, 'moon center resolution along u axis (km)', test, **options)
        test = bp.center_resolution(moon, 'v')
        show_info(bp, 'moon center resolution along v axis (km)', test, **options)
        test = bp.finest_resolution(moon)
        show_info(bp, 'moon finest resolution (km)', test, **options)
        test = bp.coarsest_resolution(moon)
        show_info(bp, 'moon coarsest resolution (km)', test, **options)

    if ring is not None:
        test = bp.resolution(ring, 'u')
        show_info(bp, 'Ring resolution along u axis (km)', test, **options)
        test = bp.resolution(ring, 'v')
        show_info(bp, 'Ring resolution along v axis (km)', test, **options)
        test = bp.center_resolution(ring, 'u')
        show_info(bp, 'Ring center resolution along u axis (km)', test, **options)
        test = bp.center_resolution(ring, 'v')
        show_info(bp, 'Ring center resolution along v axis (km)', test, **options)
        test = bp.finest_resolution(ring)
        show_info(bp, 'Ring finest resolution (km)', test, **options)
        test = bp.coarsest_resolution(ring)
        show_info(bp, 'Ring coarsest resolution (km)', test, **options)

#===========================================================================
def exercise_ansa(bp,
                  planet=None, moon=None, ring=None,
                  undersample=16, use_inventory=False, inventory_border=2,
                  **options):
    """generic unit tests for resolution.py"""

    if planet is not None:

        if planet is not None and Body.lookup(planet).ring_body is not None:
            test = bp.resolution(planet+':ansa', 'u')
            show_info(bp, 'Ansa resolution along u axis (km)', test, **options)
            test = bp.resolution(planet+':ansa', 'v')
            show_info(bp, 'Ansa resolution along v axis (km)', test, **options)
            test = bp.center_resolution(planet+':ansa', 'u')
            show_info(bp, 'Ansa center resolution along u axis (km)', test, **options)
            test = bp.center_resolution(planet+':ansa', 'v')
            show_info(bp, 'Ansa center resolution along v axis (km)', test, **options)
            test = bp.finest_resolution(planet+':ansa')
            show_info(bp, 'Ansa finest resolution (km)', test, **options)
            test = bp.coarsest_resolution(planet+':ansa')
            show_info(bp, 'Ansa coarsest resolution (km)', test, **options)


#===========================================================================
def exercise_limb(bp,
                  planet=None, moon=None, ring=None,
                  undersample=16, use_inventory=False, inventory_border=2,
                  **options):
    """generic unit tests for resolution.py"""

    if planet is not None:
        test = bp.resolution(planet+':limb', 'u')
        show_info(bp, 'Limb resolution along u axis (km)', test, **options)
        test = bp.resolution(planet+':limb', 'v')
        show_info(bp, 'Limb resolution along v axis (km)', test, **options)
        test = bp.resolution(planet+':limb', 'u')
        show_info(bp, 'Limb resolution along u axis (km)', test, **options)
        test = bp.resolution(planet+':limb', 'v')
        show_info(bp, 'Limb resolution along v axis (km)', test, **options)
        test = bp.finest_resolution(planet+':limb')
        show_info(bp, 'Limb finest resolution (km)', test, **options)
        test = bp.coarsest_resolution(planet+':limb')
        show_info(bp, 'Limb coarsest resolution (km)', test, **options)


#*******************************************************************************
class Test_Resolution(unittest.TestCase):

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
