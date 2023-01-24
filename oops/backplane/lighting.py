################################################################################
# oops/backplanes/lighting.py: Lighting geometry backplanes
################################################################################

from polymath       import Boolean, Scalar, Vector3
from oops.backplane import Backplane

def incidence_angle(self, event_key, apparent=True):
    """Incidence angle of the arriving photons at the local surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('incidence_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, arrivals=True)
    incidence = event.incidence_angle(apparent=apparent, derivs=self.ALL_DERIVS)

    # Ring incidence angles should always be 0 to pi/2
    if event.surface.COORDINATE_TYPE == 'polar':

        # Save this as the "prograde" ring incidence angle
        ring_key = ('ring_incidence_angle', event_key, 'prograde', apparent)
        self.register_backplane(ring_key, incidence)

        # _ring_flip is True wherever incidence angle has to be replaced by
        # PI - incidence. This is needed by the emission_angle backplane.
        flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
        flip_key = ('_ring_flip', event_key)
        self.register_backplane(flip_key, flip)

        # Now flip incidence angles where necessary
        if flip.any():
            incidence = Scalar.PI * flip + (1 - 2*flip) * incidence

    return self.register_backplane(key, incidence)

#===============================================================================
def emission_angle(self, event_key, apparent=True):
    """Emission angle of the departing photons at the local surface.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('emission_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key)
    emission = event.emission_angle(apparent=apparent, derivs=self.ALL_DERIVS)

    # Ring emission angles are always measured from the lit side normal
    if event.surface.COORDINATE_TYPE == 'polar':

        # Save this as the "prograde" ring incidence angle
        ring_key = ('ring_emission_angle', event_key, 'prograde', apparent)
        self.register_backplane(ring_key, emission)

        # Get the "ring flip" flag
        _ = self.incidence_angle(event_key)
        flip_key = ('_ring_flip', event_key)
        flip = self.get_backplane(flip_key)

        # Now flip emission angles where necessary
        if flip.any():
            emission = Scalar.PI * flip + (1 - 2*flip) * emission

    return self.register_backplane(key, emission)

#===============================================================================
def phase_angle(self, event_key, apparent=True):
    """Phase angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('phase_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, arrivals=True)
    phase = event.phase_angle(apparent=apparent, derivs=self.ALL_DERIVS)
    return self.register_backplane(key, phase)

#===============================================================================
def scattering_angle(self, event_key, apparent=True):
    """Scattering angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
        apparent        True for the apparent angle in the surface frame;
                        False for the actual.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('scattering_angle', event_key, apparent)
    if key in self.backplanes:
        return self.get_backplane(key)

    return self.register_backplane(key, Scalar.PI -
                                        self.phase_angle(event_key, apparent))

#===============================================================================
def center_incidence_angle(self, event_key, apparent=True):
    """Gridless incidence angle of the arriving photons at the body's central
    path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.incidence_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_emission_angle(self, event_key, apparent=True):
    """Gridless emission angle of the departing photons at the body's central
    path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.emission_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_phase_angle(self, event_key, apparent=True):
    """Gridless phase angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.phase_angle(gridless_key, apparent=apparent)

#===============================================================================
def center_scattering_angle(self, event_key, apparent=True):
    """Gridless scattering angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
        apparent        True for the apparent angle in the body frame;
                        False for the actual.
    """

    gridless_key = self.gridless_event_key(event_key)
    return self.scattering_angle(gridless_key, apparent=apparent)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite

def lighting_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.ring_names:

        phase = bp.phase_angle(name)
        bpt.gmtest(phase,
                   name + ' phase angle (deg)',
                   limit=0.01, radius=1.5, method='degrees')
        bpt.compare(phase + bp.scattering_angle(name),
                    Scalar.PI,
                    name + ' phase plus scattering angle (deg)',
                    limit=1.e-15, method='degrees')

        phase = bp.center_phase_angle(name)
        bpt.gmtest(phase,
                   name + ' center phase angle (deg)',
                   limit=0.01, method='degrees')
        bpt.compare(phase + bp.center_scattering_angle(name),
                    Scalar.PI,
                    name + ' center phase plus scattering angle (deg)',
                    limit=1.e-15, method='degrees')

        bpt.gmtest(bp.incidence_angle(name),
                   name + ' incidence angle (deg)',
                   limit=0.01, radius=1.5, method='degrees')
        bpt.gmtest(bp.emission_angle(name),
                   name + ' emission angle (deg)',
                   limit=0.01, radius=1.5, method='degrees')

        if name in bpt.ring_names:
            bpt.gmtest(bp.center_incidence_angle(name),
                       name + ' center incidence angle (deg)',
                       limit=0.01, method='degrees')
            bpt.gmtest(bp.center_emission_angle(name),
                       name + ' center emission angle (deg)',
                       limit=0.01, method='degrees')

register_test_suite('lighting', lighting_test_suite)

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.constants import DPR
from oops.backplane.unittester_support import show_info

#===============================================================================
def exercise_planet(bp,
                    planet=None, moon=None, ring=None,
                    undersample=16, use_inventory=False, inventory_border=2,
                    **options):
    """generic unit tests for lighting.py"""

    if planet is not None:
        test = bp.phase_angle(planet)
        show_info(bp, 'planet phase angle (deg)', test*DPR, **options)
        test = bp.scattering_angle(planet)
        show_info(bp, 'planet scattering angle (deg)', test*DPR, **options)
        test = bp.incidence_angle(planet)
        show_info(bp, 'planet incidence angle (deg)', test*DPR, **options)
        test = bp.emission_angle(planet)
        show_info(bp, 'planet emission angle (deg)', test*DPR, **options)
        test = bp.lambert_law(planet)
        show_info(bp, 'planet as a Lambert law', test, **options)

#===============================================================================
def exercise_ring(bp,
                  planet=None, moon=None, ring=None,
                  undersample=16, use_inventory=False, inventory_border=2,
                  **options):
    """generic unis for lighting.py"""

    if ring is not None:
        test = bp.phase_angle(ring)
        show_info(bp, 'Ring phase angle (deg)', test*DPR,**options)


#*******************************************************************************
class Test_Lighting(unittest.TestCase):

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
