################################################################################
# oops/backplanes/sky.py: Sky plane (celestial coordinates) backplanes
################################################################################

import numpy as np
from polymath       import Scalar, Vector3
from oops.backplane import Backplane
from oops.frame     import Frame

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

    return self.get_backplane(key)

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

    return self.get_backplane(key)

#===============================================================================
def _fill_ra_dec(self, event_key, apparent, direction):
    """Fill internal backplanes of RA and dec."""

    if direction not in ('arr', 'dep'):
        raise ValueError('invalid photon direction: ' + direction)

    if not event_key:
        event = self.get_obs_event(event_key)
    else:
        event = self.get_surface_event(event_key, arrivals=True)

    (ra, dec) = event.ra_and_dec(apparent=apparent, subfield=direction,
                                 derivs=self.ALL_DERIVS)
    etc = (event_key, apparent, direction)
    self.register_backplane(('right_ascension',) + etc, ra)
    self.register_backplane(('declination',)     + etc, dec)

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
        return self.get_backplane(key)

    temp_key = ('_dlos_ddec', event_key)
    if temp_key not in self.backplanes:
        self._fill_dlos_dradec(event_key)

    dlos_ddec = self.get_backplane(temp_key)
    duv_ddec = self.duv_dlos.chain(dlos_ddec)
    return self.register_backplane(key, duv_ddec.angle())

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
        return self.get_backplane(key)

    temp_key = ('_dlos_dra', event_key)
    if temp_key not in self.backplanes:
        self._fill_dlos_dradec(event_key)

    dlos_dra = self.get_backplane(temp_key)
    duv_dra = self.duv_dlos.chain(dlos_dra)
    return self.register_backplane(key, duv_dra.angle())

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

    # Convert to column vectors and save
    (dlos_dra, dlos_ddec) = dlos_dradec.extract_denoms()

    self.register_backplane(('_dlos_dra',  event_key), dlos_dra)
    self.register_backplane(('_dlos_ddec', event_key), dlos_ddec)

#===============================================================================
def center_right_ascension(self, event_key, apparent=True, direction='arr'):
    """Gridless right ascension of a photon from the body center to the
    detector.

    Input:
        event_key       key defining the event at the body's path.
        apparent        True to return the apparent direction of photons in the
                        the frame of the event; False to return the purely
                        geometric directions of the photons.
        direction       'arr' to return the direction of an arriving photon;
                        'dep' to return the direction of a departing photon.
    """

    gridless_key = self.gridless_event_key(event_key)
    key = ('center_right_ascension', gridless_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_center_ra_dec(gridless_key, apparent, direction)

    return self.get_backplane(key)

#===============================================================================
def center_declination(self, event_key, apparent=True, direction='arr'):
    """Gridless declination of a photon from the body center to the detector.

    Input:
        event_key       key defining the event at the body's path.
        apparent        True to return the apparent direction of photons in
                        the frame of the event; False to return the purely
                        geometric directions of the photons.
        direction       'arr' to return the direction of an arriving photon;
                        'dep' to return the direction of a departing photon.
    """

    gridless_key = self.gridless_event_key(event_key)
    key = ('center_declination', gridless_key, apparent, direction)
    if key not in self.backplanes:
        self._fill_center_ra_dec(gridless_key, apparent, direction)

    return self.get_backplane(key)

#===============================================================================
def _fill_center_ra_dec(self, event_key, apparent, direction):
    """Internal method to fill in RA and dec for the center of a body."""

    if direction not in ('arr', 'dep'):
        raise ValueError('invalid photon direction: ' + direction)

    gridless_key = self.gridless_event_key(event_key)
    event = self.get_obs_event(gridless_key)
    (ra, dec) = event.ra_and_dec(apparent=apparent, subfield=direction,
                                 derivs=self.ALL_DERIVS)
    etc = (gridless_key, apparent, direction)
    self.register_backplane(('center_right_ascension',) + etc, ra)
    self.register_backplane(('center_declination',)     + etc, dec)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite
from oops.constants import HALFPI, DPR

def sky_test_suite(bpt):

    bp = bpt.backplane

    # Right ascension
    cos_dec = bp.declination().cos().mean(builtins=True)
    actual = bp.right_ascension(apparent=False)
    apparent = bp.right_ascension(apparent=True)
    bpt.gmtest(actual,
               'Right ascension (deg, actual)',
               limit=1.e-6/cos_dec, method='mod360', radius=1.)
    bpt.gmtest(apparent,
               'Right ascension (deg, apparent)',
               limit=1.e-6/cos_dec, method='mod360', radius=1.)
    bpt.compare(actual - apparent, 0.,
                'Right ascension, actual minus apparent (deg)',
                limit=0.1/cos_dec, method='mod360')

    # Declination
    actual = bp.declination(apparent=False)
    apparent = bp.declination(apparent=True)
    bpt.gmtest(actual,
               'Declination (deg, actual)',
               limit=1.e-6, method='degrees', radius=1.)
    bpt.gmtest(apparent,
               'Declination (deg, apparent)',
               limit=1.e-6, method='degrees', radius=1.)
    bpt.compare(actual - apparent, 0.,
                'Declination, actual minus apparent (deg)',
                limit=0.1/cos_dec, method='mod360')

    # Sky angles
    north = bp.celestial_north_angle()
    east  = bp.celestial_east_angle()
    bpt.gmtest(north,
               'Celestial north angle (deg)',
               method='mod360', limit=0.001)
    bpt.gmtest(east,
               'Celestial east angle (deg)',
               method='mod360', limit=0.001)
    bpt.compare(north - east, HALFPI,
                'Celestial north minus east angles (deg)',
                method='mod360', limit=2.)

    for name in bpt.body_names:

        # Right ascension
        cos_dec = bp.center_declination(name).cos().mean(builtins=True)
        actual = bp.center_right_ascension(name, apparent=False)
        apparent = bp.center_right_ascension(name, apparent=True)
        bpt.gmtest(actual,
                   name + ' center right ascension (deg, actual)',
                   limit=1.e-6/cos_dec, method='mod360')
        bpt.gmtest(apparent,
                   name + ' center right ascension (deg, apparent)',
                   limit=1.e-6/cos_dec, method='mod360')
        bpt.compare(actual - apparent, 0.,
                    name + ' center right ascension, actual minus apparent (deg)',
                    limit=0.1/cos_dec, method='mod360')

        # Declination
        actual = bp.center_declination(name, apparent=False)
        apparent = bp.center_declination(name, apparent=True)
        bpt.gmtest(actual,
                   name + ' center declination (deg, actual)',
                   limit=1.e-6, method='degrees')
        bpt.gmtest(apparent,
                   name + ' center declination (deg, apparent)',
                   limit=1.e-6, method='degrees')
        bpt.compare(actual - apparent, 0.,
                    name + ' center declination, actual minus apparent (deg)',
                    limit=0.1, method='degrees')

    # Derivative tests
    if bpt.derivs:
        (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
        pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)
        cos_dec = bp.declination().cos().mean(builtins=True)
        (ulimit, vlimit) = DPR * pixel_duv * 1.e-4

        # right_ascension
        ra = bp.right_ascension()
        dra_duv = ra.d_dlos.chain(bp.dlos_duv)
        (dra_du, dra_dv) = dra_duv.extract_denoms()

        dra = bp_u1.right_ascension() - bp_u0.right_ascension()
        dra = Scalar.PI - (dra.wod - Scalar.PI).abs()
        bpt.compare(dra/bpt.duv, dra_du,
                    'Right ascension d/du self-check (deg/pix)',
                    limit=ulimit/cos_dec, radius=1, method='degrees')

        dra = bp_v1.right_ascension() - bp_v0.right_ascension()
        dra = Scalar.PI - (dra.wod - Scalar.PI).abs()
        bpt.compare(dra/bpt.duv, dra_dv,
                    'Right ascension d/dv self-check (deg/pix)',
                    limit=vlimit/cos_dec, radius=1, method='degrees')

        # declination
        dec = bp.declination()
        ddec_duv = dec.d_dlos.chain(bp.dlos_duv)
        (ddec_du, ddec_dv) = ddec_duv.extract_denoms()

        ddec = bp_u1.declination() - bp_u0.declination()
        bpt.compare(ddec.wod/bpt.duv, ddec_du,
                    'Declination d/du self-check (deg/pix)',
                    limit=ulimit, radius=1, method='degrees')

        ddec = bp_v1.declination() - bp_v0.declination()
        bpt.compare(ddec.wod/bpt.duv, ddec_dv,
                    'Declination d/dv self-check (deg/pix)',
                    limit=vlimit, radius=1, method='degrees')

register_test_suite('sky', sky_test_suite)

################################################################################
