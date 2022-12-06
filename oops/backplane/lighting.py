################################################################################
# oops/backplanes/lighting.py: Lighting geometry backplanes
################################################################################

from polymath       import Boolean, Scalar
from oops.backplane import Backplane

#===============================================================================
def incidence_angle(self, event_key):
    """Incidence angle of the arriving photons at the local surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('incidence_angle', event_key)
    if key not in self.backplanes:
        event = self.get_surface_event_with_arr(event_key)
        incidence = event.incidence_angle()

        # Ring incidence angles are always 0-90 degrees
        if event.surface.COORDINATE_TYPE == 'polar':

            # flip is True wherever incidence angle has to be changed
            flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
            self.register_backplane(('ring_flip', event_key), flip)

            # Now flip incidence angles where necessary
            incidence = Scalar.PI * flip + (1. - 2.*flip) * incidence

        self.register_backplane(key, incidence)

    return self.backplanes[key]

#===============================================================================
def emission_angle(self, event_key):
    """Emission angle of the departing photons at the local surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('emission_angle', event_key)
    if key not in self.backplanes:
        event = self.get_surface_event(event_key)
        emission = event.emission_angle()

        # Ring emission angles are always measured from the lit side normal
        if event.surface.COORDINATE_TYPE == 'polar':

            # Get the flip flag
            _ = self.incidence_angle(event_key)
            flip = self.backplanes[('ring_flip', event_key)]

            # Flip emission angles where necessary
            emission = Scalar.PI * flip + (1. - 2.*flip) * emission

        self.register_backplane(key, emission)

    return self.backplanes[key]

#===============================================================================
def phase_angle(self, event_key):
    """Phase angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('phase_angle', event_key)
    if key not in self.backplanes:
        event = self.get_surface_event_with_arr(event_key)
        self.register_backplane(key, event.phase_angle())

    return self.backplanes[key]

#===============================================================================
def scattering_angle(self, event_key):
    """Scattering angle between the arriving and departing photons.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('scattering_angle', event_key)
    if key not in self.backplanes:
        self.register_backplane(key, Scalar.PI -
                                     self.phase_angle(event_key))

    return self.backplanes[key]

#===============================================================================
def center_incidence_angle(self, event_key):
    """Incidence angle of the arriving photons at the body's central path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('center_incidence_angle', event_key)
    if key not in self.backplanes:
        event = self.get_gridless_event_with_arr(event_key)

        # Sign on event.arr is negative because photon is incoming
        latitude = (event.neg_arr_ap.to_scalar(2) /
                    event.arr_ap.norm()).arcsin()
        incidence = Scalar.HALFPI - latitude

        # Ring incidence angles are always 0-90 degrees
        if event.surface.COORDINATE_TYPE == 'polar':

            # The flip is True wherever incidence angle has to be changed
            flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
            self.register_gridless_backplane(('ring_center_flip',
                                              event_key), flip)

            # Now flip incidence angles where necessary
            if flip.any():
                incidence = Scalar.PI * flip + (1. - 2.*flip) * incidence

        self.register_gridless_backplane(key, incidence)

    return self.backplanes[key]

#===============================================================================
def center_emission_angle(self, event_key):
    """Emission angle of the departing photons at the body's central path.

    This uses the z-axis of the body's frame to define the local normal.

    Input:
        event_key       key defining the event on the body's path.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('center_emission_angle', event_key)
    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)

        latitude = (event.dep_ap.to_scalar(2) /
                    event.dep_ap.norm()).arcsin()
        emission = Scalar.HALFPI - latitude

        # Ring emission angles are always measured from the lit side normal
        if event.surface.COORDINATE_TYPE == 'polar':

            # Get the flip flag
            _ = self.center_incidence_angle(event_key)
            flip = self.backplanes[('ring_center_flip', event_key)]

            # Flip emission angles where necessary
            if flip.any():
                emission = Scalar.PI * flip + (1. - 2.*flip) * emission

        self.register_gridless_backplane(key, emission)

    return self.backplanes[key]

#===============================================================================
def center_phase_angle(self, event_key):
    """Phase angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('center_phase_angle', event_key)
    if key not in self.backplanes:
        event = self.get_gridless_event_with_arr(event_key)
        self.register_gridless_backplane(key, event.phase_angle())

    return self.backplanes[key]

#===============================================================================
def center_scattering_angle(self, event_key):
    """Scattering angle as measured at the body's central path.

    Input:
        event_key       key defining the event on the body's path.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('center_scattering_angle', event_key)
    if key not in self.backplanes:
        angle = Scalar.PI - self.center_phase_angle(event_key)
        self.register_gridless_backplane(key, angle)

    return self.backplanes[key]

################################################################################

Backplane._define_backplane_names(globals().copy())

################################################################################


################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.meshgrid                     import Meshgrid
from oops.unittester_support           import TESTDATA_PARENT_DIRECTORY
from oops.constants                    import DPR
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
