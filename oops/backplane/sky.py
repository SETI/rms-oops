################################################################################
# oops/backplanes/sky.py: Sky plane (celestial coordinates) backplanes
################################################################################

import numpy as np
from polymath import Scalar, Vector3

from oops.backplane import Backplane
from oops.frame     import Frame

#===============================================================================
def right_ascension(self, event_key=(), apparent=True, direction='arr'):
    """Right ascension of the arriving or departing photon

    Optionally, it allows for stellar aberration.

    Input:
        event_key       key defining the surface event, typically () to refer to
                        the observation.
        apparent        True to return the apparent direction of photons in the
                        frame of the event; False to return the purely geometric
                        directions of the photons.
        direction       'arr' to return the direction of an arriving photon;
                        'dep' to return the direction of a departing photon.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('right_ascension', event_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_ra_dec(event_key, apparent, direction)

    return self.backplanes[key]

#===============================================================================
def declination(self, event_key=(), apparent=True, direction='arr'):
    """Declination of the arriving or departing photon.

    Optionally, it allows for stellar aberration.

    Input:
        event_key       key defining the surface event, typically () to refer to
                        the observation.
        apparent        True to return the apparent direction of photons in the
                        frame of the event; False to return the purely geometric
                        directions of the photons.
        direction       'arr' to base the direction on an arriving photon;
                        'dep' to base the direction on a departing photon.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('declination', event_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_ra_dec(event_key, apparent, direction)

    return self.backplanes[key]

#===============================================================================
def _fill_ra_dec(self, event_key, apparent, direction):
    """Fill internal backplanes of RA and dec."""

    assert direction in ('arr', 'dep')

    event = self.get_surface_event_with_arr(event_key)
    (ra, dec) = event.ra_and_dec(apparent=apparent, subfield=direction)

    self.register_backplane(('right_ascension', event_key, apparent, direction),
                            ra)
    self.register_backplane(('declination', event_key, apparent, direction),
                            dec)

#===============================================================================
def celestial_north_angle(self, event_key=()):
    """Direction of celestial north at each pixel in the image.

    The angle is measured from the U-axis toward the V-axis. This varies across
    the field of view due to spherical distortion and also any distortion in the
    FOV.

    Input:
        event_key       key defining the surface event, typically () to refer
                        refer to the observation.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('celestial_north_angle', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    temp_key = ('dlos_ddec', event_key)
    if temp_key not in self.backplanes:
        self._fill_dlos_dradec(event_key)

    dlos_ddec = self.backplanes[temp_key]
    duv_ddec = self.duv_dlos.chain(dlos_ddec)

    du_ddec_vals = duv_ddec.vals[...,0]
    dv_ddec_vals = duv_ddec.vals[...,1]
    clock = np.arctan2(dv_ddec_vals, du_ddec_vals)

    self.register_backplane(key, Scalar(clock, duv_ddec.mask))

    return self.backplanes[key]

#===============================================================================
def celestial_east_angle(self, event_key=()):
    """Direction of celestial north at each pixel in the image.

    The angle is measured from the U-axis toward the V-axis. This varies
    across the field of view due to spherical distortion and also any
    distortion in the FOV.

    Input:
        event_key       key defining the surface event, typically () to
                        refer to the observation.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('celestial_east_angle', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    temp_key = ('dlos_dra', event_key)
    if temp_key not in self.backplanes:
        self._fill_dlos_dradec(event_key)

    dlos_dra = self.backplanes[temp_key]
    duv_dra = self.duv_dlos.chain(dlos_dra)

    du_dra_vals = duv_dra.vals[...,0]
    dv_dra_vals = duv_dra.vals[...,1]
    clock = np.arctan2(dv_dra_vals, du_dra_vals)
    self.register_backplane(key, Scalar(clock, duv_dra.mask))

    return self.backplanes[key]

#===============================================================================
def _fill_dlos_dradec(self, event_key):
    """Fill internal backplanes with derivatives with respect to RA and dec.
    """

    ra = self.right_ascension(event_key)
    dec = self.declination(event_key)

    # Derivatives of...
    #   los[0] = cos(dec) * cos(ra)
    #   los[1] = cos(dec) * sin(ra)
    #   los[2] = sin(dec)
    cos_dec = np.cos(dec.vals)
    sin_dec = np.sin(dec.vals)

    cos_ra = np.cos(ra.vals)
    sin_ra = np.sin(ra.vals)

    dlos_dradec_vals = np.zeros(ra.shape + (3,2))
    dlos_dradec_vals[...,0,0] = -sin_ra * cos_dec
    dlos_dradec_vals[...,1,0] =  cos_ra * cos_dec
    dlos_dradec_vals[...,0,1] = -sin_dec * cos_ra
    dlos_dradec_vals[...,1,1] = -sin_dec * sin_ra
    dlos_dradec_vals[...,2,1] =  cos_dec

    dlos_dradec_j2000 = Vector3(dlos_dradec_vals, ra.mask, drank=1)

    # Rotate dlos from the J2000 frame to the image coordinate frame
    frame = self.obs.frame.wrt(Frame.J2000)
    xform = frame.transform_at_time(self.obs_event.time)

    dlos_dradec = xform.rotate(dlos_dradec_j2000)

    # Convert to a column matrix and save
    dlos_dra  = Vector3(dlos_dradec.vals[...,0], ra.mask)
    dlos_ddec = Vector3(dlos_dradec.vals[...,1], ra.mask)

    self.register_backplane(('dlos_dra',  event_key), dlos_dra)
    self.register_backplane(('dlos_ddec', event_key), dlos_ddec)

#===============================================================================
def center_right_ascension(self, event_key, apparent=True, direction='arr'):
    """Right ascension of the arriving or departing photon

    Optionally, it allows for stellar aberration and for frames other than
    J2000.

    Input:
        event_key       key defining the event at the body's path.
        apparent        True to return the apparent direction of photons in the
                        the frame of the event; False to return the purely
                        geometric directions of the photons.
        direction       'arr' to return the direction of an arriving photon;
                        'dep' to return the direction of a departing photon.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('center_right_ascension', event_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_center_ra_dec(event_key, apparent, direction)

    return self.backplanes[key]

#===============================================================================
def center_declination(self, event_key, apparent=True, direction='arr'):
    """Declination of the arriving or departing photon.

    Optionally, it allows for stellar aberration and for frames other than
    J2000.

    Input:
        event_key       key defining the event at the body's path.
        apparent        True to return the apparent direction of photons in
                        the frame of the event; False to return the purely
                        geometric directions of the photons.
        direction       'arr' to return the direction of an arriving photon;
                        'dep' to return the direction of a departing photon.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('center_declination', event_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_center_ra_dec(event_key, apparent, direction)

    return self.backplanes[key]

#===============================================================================
def _fill_center_ra_dec(self, event_key, apparent, direction):
    """Internal method to fill in RA and dec for the center of a body."""

    assert direction in ('arr', 'dep')

    _ = self.get_gridless_event_with_arr(event_key)
    event = self.gridless_arrivals[event_key]

    (ra, dec) = event.ra_and_dec(apparent, subfield=direction)

    self.register_gridless_backplane(
            ('center_right_ascension', event_key, apparent, direction), ra)
    self.register_gridless_backplane(
            ('center_declination', event_key, apparent, direction), dec)

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
def exercise_right_ascension(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for sky.py"""
    
    test = bp.right_ascension(apparent=False)
    show_info('Right ascension (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    test = bp.right_ascension(apparent=True)
    show_info('Right ascension (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if planet != None:
        test = bp.center_right_ascension(planet, apparent=False)
        show_info('Right ascension of planet (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_right_ascension(planet, apparent=True)
        show_info('Right ascension of planet (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.center_right_ascension(moon, apparent=False)
        show_info('Right ascension of moon (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_right_ascension(moon, apparent=True)
        show_info('Right ascension of moon (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_declination(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for sky.py"""
    
    test = bp.declination(apparent=False)
    show_info('Declination (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    test = bp.declination(apparent=True)
    show_info('Declination (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if planet != None:
        test = bp.center_declination(planet, apparent=False)
        show_info('Declination of planet (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_declination(planet, apparent=True)
        show_info('Declination of planet (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if moon != None:
        test = bp.center_declination(moon, apparent=False)
        show_info('Declination of moon (deg, actual)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.center_declination(moon, apparent=True)
        show_info('Declination of moon (deg, apparent)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_celestial_and_polar_angles(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for sky.py"""
    
    test = bp.celestial_north_angle()
    show_info('Celestial north angle (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    test = bp.celestial_east_angle()
    show_info('Celestial east angle (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)



#*******************************************************************************
class Test_Sky(unittest.TestCase):

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
