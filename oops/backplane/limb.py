################################################################################
# oops/backplanes/limb_backplanes.py: Latitude/longitude backplanes
################################################################################

from oops.backplane import Backplane

#===============================================================================
def limb_altitude(self, event_key, limit=None, lock_limits=False):
    """Elevation of a limb point above the body's surface.

    Input:
        event_key       key defining the ring surface event.
        limit           upper limit to altitude in km. Higher altitudes are
                        masked.
        lock_limits     if True, the limit will be applied to the default event,
                        so that all backplanes generated from this event_key
                        will have the same upper limit. This can only be applied
                        the first time this event_key is used.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('limb_altitude', event_key, limit)
    if key in self.backplanes:
        return self.backplanes[key]

    self._fill_limb_intercepts(event_key, limit, lock_limits)
    return self.backplanes[key]

#===============================================================================
def altitude(self, event_key, limit=None):
    """Deprecated name for limb_altitude()."""

    return self.limb_altitude(event_key)

#===============================================================================
def _fill_limb_intercepts(self, event_key, limit=None, lock_limits=False):
    """Internal method to fill in the limb intercept geometry backplanes.

    Input:
        event_key       key defining the ring surface event.
        limit           upper limit to altitude in km. Higher altitudes are
                        masked.
        lock_limits     if True, the limit will be applied to the default event,
                        so that all backplanes generated from this event_key
                        will share the same limit. This can only be applied the
                        first time this event_key is used.
    """

    # Don't allow lock_limits if the backplane was already generated
    if limit is None:
        lock_limits = False

    if lock_limits and event_key in self.surface_events:
        raise ValueError('lock_limits option disallowed for pre-existing ' +
                         'limb event key ' + str(event_key))

    # Get the limb intercept coordinates
    event = self.get_surface_event(event_key)
    if event.surface.COORDINATE_TYPE != 'limb':
        raise ValueError('limb intercepts require a "limb" surface type')

    # Limit the event if necessary
    if lock_limits:

        # Apply the upper limit to the event
        altitude = event.coord3
        self.apply_mask_to_event(event_key, altitude > limit)
        event = self.get_surface_event(event_key)

    # Register the default backplanes
    self.register_backplane(('longitude', event_key, 'iau', 'east', 0,
                             'squashed'), event.coord1)
    self.register_backplane(('latitude', event_key, 'squashed'), event.coord2)
    self.register_backplane(('limb_altitude', event_key, None), event.coord3)

    # Apply a mask just to this backplane if necessary
    if limit is not None:
        altitude = event.coord3.mask_where_gt(limit)
        self.register_backplane(('limb_altitude', event_key, limit), altitude)

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
def exercise(bp, printing, saving, dir, refdir,
             planet=None, moon=None, ring=None,
             undersample=16, use_inventory=False, inventory_border=2):
    """generic unit tests for limb.py"""

    if planet != None:
        test = bp.limb_altitude(planet+':limb')
        show_info(bp, 'Limb altitude (km)', test,
                     printing=printing, saving=saving, dir=dir, refdir=refdir)



#*******************************************************************************
class Test_Limb(unittest.TestCase):

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
