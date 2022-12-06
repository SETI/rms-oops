################################################################################
# oops/backplanes/spheroid.py: Spheroid/Ellipsoid backplanes
################################################################################

from __future__ import print_function

from polymath       import Scalar
from oops.backplane import Backplane
from oops.constants import PI, TWOPI

################################################################################
# Body surface geometry, surface intercept versions
#   longitude()
#   latitude()
################################################################################

#===============================================================================
def longitude(self, event_key, reference='iau', direction='west',
                               minimum=0, lon_type='centric'):
    """Longitude at the surface intercept point in the image.

    Input:
        event_key       key defining the surface event.
        reference       defines the location of zero longitude.
                        'iau' for the IAU-defined prime meridian;
                        'obs' for the sub-observer longitude;
                        'sun' for the sub-solar longitude;
                        'oha' for the anti-observer longitude;
                        'sha' for the anti-solar longitude, returning the
                              local time on the planet if direction is west.
        direction       direction on the surface of increasing longitude,
                        'east' or 'west'.
        minimum         the smallest numeric value of longitude, either 0
                        or -180.
        lon_type        defines the type of longitude measurement:
                        'centric'   for planetocentric;
                        'graphic'   for planetographic;
                        'squashed'  for an intermediate longitude type used
                                    internally.
                        Note that lon_type is irrelevant to Spheroids but
                        matters for Ellipsoids.
    """

    event_key = self.standardize_event_key(event_key)
    assert reference in ('iau', 'sun', 'sha', 'obs', 'oha')
    assert direction in ('east', 'west')
    assert minimum in (0, -180)
    assert lon_type in ('centric', 'graphic', 'squashed')

    # Look up under the desired reference
    key0 = ('longitude', event_key)
    key = key0 + (reference, direction, minimum, lon_type)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with default keys, fill in those backplanes
    # Note that longitudes default to eastward for right-handed
    # coordinates
    key_default = key0 + ('iau', 'east', 0, 'squashed')
    if key_default not in self.backplanes:
        self._fill_surface_intercepts(event_key)

    # Fill in the required longitude type if necessary
    key_typed = key0 + ('iau', 'east', 0, lon_type)
    if key_typed in self.backplanes:
        lon = self.backplanes[key_typed]
    else:
        lon_squashed = self.backplanes[key_default]
        surface = self.get_surface(event_key)

        if lon_type == 'centric':
            lon = surface.lon_to_centric(lon_squashed)
            self.register_backplane(key_typed, lon)
        else:
            lon = surface.lon_to_graphic(lon_squashed)
            self.register_backplane(key_typed, lon)

    # Define the longitude relative to the reference value
    if reference != 'iau':
        if reference in ('sun', 'sha'):
            ref_lon = self._sub_solar_longitude(event_key)
        else:
            ref_lon = self._sub_observer_longitude(event_key)

        if reference in ('sha', 'oha'):
            ref_lon = ref_lon - Scalar.PI

        lon = lon - ref_lon

    # Reverse if necessary
    if direction == 'west':
        lon = -lon

    # Re-define the minimum
    if minimum == 0:
        lon = lon % TWOPI
    else:
        lon = (lon + PI) % TWOPI - Scalar.PI

    self.register_backplane(key, lon)
    return self.backplanes[key]

#===============================================================================
def latitude(self, event_key, lat_type='centric'):
    """Latitude at the surface intercept point in the image.

    Input:
        event_key       key defining the surface event.
        lat_type        defines the type of latitude measurement:
                        'centric'   for planetocentric;
                        'graphic'   for planetographic;
                        'squashed'  for an intermediate latitude type used
                                    internally.
    """

    event_key = self.standardize_event_key(event_key)
    assert lat_type in ('centric', 'graphic', 'squashed')

    # Look up under the desired reference
    key0 = ('latitude', event_key)
    key = key0 + (lat_type,)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with default keys, fill in those backplanes
    key_default = key0 + ('squashed',)
    if key_default not in self.backplanes:
        self._fill_surface_intercepts(event_key)

    # Fill in the values for this key
    lat = self.backplanes[key_default]
    if lat_type == 'squashed':
        return lat

    surface = self.get_surface(event_key)

    # Fill in the requested lon_type if necessary
    lon_key = ('longitude', event_key, 'iau', 'east', 0, 'squashed')
    lon = self.backplanes[lon_key]

    if lat_type == 'centric':
        lat = surface.lat_to_centric(lat, lon)
    else:
        lat = surface.lat_to_graphic(lat, lon)

    self.register_backplane(key, lat)
    return self.backplanes[key]

#===============================================================================
def _fill_surface_intercepts(self, event_key):
    """Internal method to fill in the surface intercept geometry backplanes.
    """

    # Get the surface intercept coordinates
    event_key = self.standardize_event_key(event_key)
    event = self.get_surface_event(event_key)

    # If this is actually a limb event, define the limb backplanes instead
    if event.surface.COORDINATE_TYPE == 'limb':
        self._fill_limb_intercepts(event_key)
        return

    lon_key = ('longitude', event_key, 'iau', 'east', 0, 'squashed')
    lat_key = ('latitude', event_key, 'squashed')

    assert event.surface.COORDINATE_TYPE == 'spherical'

    self.register_backplane(lon_key, event.coord1)
    self.register_backplane(lat_key, event.coord2)

#===============================================================================
def _sub_observer_longitude(self, event_key):
    """Sub-observer longitude. Used internally."""

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_observer_longitude', event_key)

    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)
        dep_ap = event.apparent_dep()       # for ABERRATION=old or new
        lon = dep_ap.to_scalar(1).arctan2(dep_ap.to_scalar(0)) % TWOPI

        self.register_gridless_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def _sub_observer_latitude(self, event_key):
    """Sub-observer latitude. Used internally."""

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_observer_latitude', event_key)

    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)
        dep_ap = event.apparent_dep()       # for ABERRATION=old or new
        lat = (dep_ap.to_scalar(2) / dep_ap.norm()).arcsin()

        self.register_gridless_backplane(key, lat)

    return self.backplanes[key]

#===============================================================================
def _sub_solar_longitude(self, event_key):
    """Sub-solar longitude. Used internally."""

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_solar_longitude', event_key)

    if key not in self.backplanes:
        event = self.get_gridless_event_with_arr(event_key)
        neg_arr_ap = -event.apparent_arr()  # for ABERRATION=old or new
        lon = neg_arr_ap.to_scalar(1).arctan2(neg_arr_ap.to_scalar(0)) % \
                                                                       TWOPI

        self.register_gridless_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def _sub_solar_latitude(self, event_key):
    """Sub-solar latitude. Used internally."""

    event_key = self.standardize_event_key(event_key)
    key = ('_sub_solar_latitude', event_key)

    if key not in self.backplanes:
        event = self.get_gridless_event_with_arr(event_key)
        neg_arr_ap = -event.apparent_arr()  # for ABERRATION=old or new
        lat = (neg_arr_ap.to_scalar(2) / neg_arr_ap.norm()).arcsin()

        self.register_gridless_backplane(key, lat)

    return self.backplanes[key]

################################################################################
# Surface geometry, path intercept versions
#   sub_observer_longitude()
#   sub_solar_longitude()
#   sub_observer_latitude()
#   sub_solar_latitude()
################################################################################

#===============================================================================
def sub_observer_longitude(self, event_key, reference='iau',
                                 direction='west', minimum=0):
    """Sub-observer longitude.

    Input:
        event_key   key defining the surface event.
        reference   defines the location of zero longitude.
                    'iau' for the IAU-defined prime meridian;
                    'obs' for the sub-observer longitude;
                    'sun' for the sub-solar longitude;
                    'oha' for the anti-observer longitude;
                    'sha' for the anti-solar longitude, returning the
                          local time on the planet if direction is west.
        direction   direction on the surface of increasing longitude, 'east'
                    or 'west'.
        minimum     the smallest numeric value of longitude, either 0 or
                    -180.
    """

    key0 = ('sub_observer_longitude', event_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.backplanes[key]

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        lon = self.backplanes[key_default]
    else:
        lon = self._sub_observer_longitude(event_key)
        self.register_gridless_backplane(key_default, lon)

    if key == key_default:
        return lon

    lon = self._sub_longitude(event_key, lon, reference, direction,
                                              minimum)

    self.register_gridless_backplane(key, lon)
    return lon

#===============================================================================
def sub_solar_longitude(self, event_key, reference='iau',
                                         direction='west', minimum=0):
    """Sub-solar longitude.

    Note that this longitude is essentially independent of the
    longitude_type (centric, graphic or squashed).

    Input:
        event_key   key defining the surface event.
        reference   defines the location of zero longitude.
                    'iau' for the IAU-defined prime meridian;
                    'obs' for the sub-observer longitude;
                    'sun' for the sub-solar longitude;
                    'oha' for the anti-observer longitude;
                    'sha' for the anti-solar longitude, returning the
                          local time on the planet if direction is west.
        direction   direction on the surface of increasing longitude, 'east'
                    or 'west'.
        minimum     the smallest numeric value of longitude, either 0 or
                    -180.
    """

    key0 = ('sub_solar_longitude', event_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.backplanes[key]

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        lon = self.backplanes[key_default]
    else:
        lon = self._sub_solar_longitude(event_key)
        self.register_gridless_backplane(key_default, lon)

    if key == key_default:
        return lon

    lon = self._sub_longitude(event_key, lon, reference, direction,
                                              minimum)

    self.register_gridless_backplane(key, lon)
    return lon

#===============================================================================
def _sub_longitude(self, event_key, lon, reference='iau',
                                         direction='west', minimum=0):
    """Sub-solar or sub-observer longitude."""

    event_key = self.standardize_event_key(event_key)
    assert reference in ('iau', 'sun', 'sha', 'obs', 'oha')
    assert direction in ('east', 'west')
    assert minimum in (0, -180)

    # Define the longitude relative to the reference value
    if reference != 'iau':
        if reference in ('sun', 'sha'):
            ref_lon = self._sub_solar_longitude(event_key)
        else:
            ref_lon = self._sub_observer_longitude(event_key)

        if reference in ('sha', 'oha'):
            ref_lon = ref_lon - Scalar.PI

        lon = lon - ref_lon

    # Reverse if necessary
    if direction == 'west':
        lon = -lon

    # Re-define the minimum
    if minimum == 0:
        lon = lon % Scalar.TWOPI
    else:
        lon = (lon + Scalar.PI) % TWOPI - Scalar.PI

    return lon

#===============================================================================
def sub_observer_latitude(self, event_key, lat_type='centric'):
    """Sub-observer latitude at the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        lat_type        "centric" for planetocentric latitude;
                        "graphic" for planetographic latitude.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('sub_observer_latitude', event_key, lat_type)
    assert lat_type in ('centric', 'graphic')

    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)
        dep_ap = event.apparent_dep()       # for ABERRATION=old or new

        if lat_type == 'graphic':
            dep_ap = dep_ap.element_mul(event.surface.unsquash_sq)

        lat = (dep_ap.to_scalar(2) / dep_ap.norm()).arcsin()

        self.register_gridless_backplane(key, lat)

    return self.backplanes[key]

#===============================================================================
def sub_solar_latitude(self, event_key, lat_type='centric'):
    """Sub-solar latitude at the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        lat_type        "centric" for planetocentric latitude;
                        "graphic" for planetographic latitude.
    """

    event_key = self.standardize_event_key(event_key)

    key = ('sub_solar_latitude', event_key, lat_type)
    assert lat_type in ('centric', 'graphic')

    if key not in self.backplanes:
        event = self.get_gridless_event_with_arr(event_key)
        neg_arr_ap = -event.apparent_arr()  # for ABERRATION=old or new

        if lat_type == 'graphic':
            neg_arr_ap = neg_arr_ap.element_mul(event.surface.unsquash_sq)

        lat = (neg_arr_ap.to_scalar(2) / neg_arr_ap.norm()).arcsin()

        self.register_gridless_backplane(key, lat)

    return self.backplanes[key]

#===============================================================================
def lambert_law(self, event_key):
    """Lambert law model cos(incidence_angle) for the surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lambert_law', event_key)
    if key not in self.backplanes:
        lambert_law = self.incidence_angle(event_key).cos()
        lambert_law = lambert_law.mask_where(lambert_law <= 0., 0.)
        self.register_backplane(key, lambert_law)

    return self.backplanes[key]

#===============================================================================
def minnaert_law(self, event_key, k, k2=None):
    """Minnaert law model for the surface.

    Input:
        event_key       key defining the surface event.
        k               The Minnaert exponent (for cos(i)).
        k2              Optional second Minnaert exponent (for cos(e)).
                        Defaults to k-1.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('minnaert_law', event_key)
    if key not in self.backplanes:
        if k2 is None:
            k2 = k-1.
        mu0 = self.lambert_law(event_key) # Masked
        mu = self.emission_angle(event_key).cos()
        minnaert_law = mu0 ** k * mu ** k2
        self.register_backplane(key, minnaert_law)

    return self.backplanes[key]

#===============================================================================
def lommel_seeliger_law(self, event_key):
    """Lommel-Seeliger law model for the surface.

    Returns mu0 / (mu + mu0)

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lommel_seeliger_law', event_key)
    if key not in self.backplanes:
        mu0 = self.incidence_angle(event_key).cos()
        mu  = self.emission_angle(event_key).cos()
        lommel_seeliger_law = mu0 / (mu + mu0)
        lommel_seeliger_law = lommel_seeliger_law.mask_where(mu0 <= 0., 0.)
        self.register_backplane(key, lommel_seeliger_law)

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
def exercise_limb_longitude(bp,
                            planet=None, moon=None, ring=None,
                            undersample=16, use_inventory=False,
                            inventory_border=2,
                            **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        test = bp.longitude(planet+':limb', 'iau')
        show_info(bp, 'Limb longitude wrt IAU (deg)', test*DPR, **options)
        test = bp.longitude(planet+':limb', 'obs')
        show_info(bp, 'Limb longitude wrt observer (deg)', test*DPR, **options)
        test = bp.longitude(planet+':limb', reference='obs', minimum=-180)
        show_info(bp, 'Limb longitude wrt observer, -180 (deg)', test*DPR, **options)
        test = bp.longitude(planet+':limb', 'oha')
        show_info(bp, 'Limb longitude wrt OHA (deg)', test*DPR, **options)
        test = bp.longitude(planet+':limb', 'sun')
        show_info(bp, 'Limb longitude wrt Sun (deg)', test*DPR, **options)
        test = bp.longitude(planet+':limb', 'sha')
        show_info(bp, 'Limb longitude wrt SHA (deg)', test*DPR, **options)

#===============================================================================
def exercise_limb_latitude(bp,
                           planet=None, moon=None, ring=None,
                           undersample=16, use_inventory=False,
                           inventory_border=2,
                           **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        test = bp.latitude(planet+':limb', lat_type='centric')
        show_info(bp, 'Limb planetocentric latitude (deg)', test*DPR, **options)
        test = bp.latitude(planet+':limb', lat_type='squashed')
        show_info(bp, 'Limb squashed latitude (deg)', test*DPR, **options)
        test = bp.latitude(planet+':limb', lat_type='graphic')
        show_info(bp, 'Limb planetographic latitude (deg)', test*DPR, **options)

#===============================================================================
def exercise_surface_latitude(bp,
                              planet=None, moon=None, ring=None,
                              undersample=16, use_inventory=False,
                              inventory_border=2,
                              **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        test = bp.latitude(planet, lat_type='centric')
        show_info(bp, 'planet latitude, planetocentric (deg)', test*DPR, **options)
        test = bp.latitude(planet, lat_type='squashed')
        show_info(bp, 'planet latitude, squashed (deg)', test*DPR, **options)
        test = bp.latitude(planet, lat_type='graphic')
        show_info(bp, 'planet latitude, planetographic (deg)', test*DPR, **options)
        test = bp.sub_observer_latitude(planet)
        show_info(bp, 'planet sub-observer latitude (deg)', test*DPR, **options)
        test = bp.sub_solar_latitude(planet)
        show_info(bp, 'planet sub-solar latitude (deg)', test*DPR, **options)

    if moon is not None:
        test = bp.latitude(moon, lat_type='centric')
        show_info(bp, 'moon latitude, planetocentric (deg)', test*DPR, **options)
        test = bp.latitude(moon, lat_type='squashed')
        show_info(bp, 'moon latitude, squashed (deg)', test*DPR, **options)
        test = bp.latitude(moon, lat_type='graphic')
        show_info(bp, 'moon latitude, planetographic (deg)', test*DPR, **options)
        test = bp.sub_observer_latitude(moon)
        show_info(bp, 'moon sub-observer latitude (deg)', test*DPR, **options)
        test = bp.sub_solar_latitude(moon)
        show_info(bp, 'moon sub-solar latitude (deg)', test*DPR, **options)

#===============================================================================
def exercise_surface_planet_moon(bp,
                                 planet=None, moon=None, ring=None,
                                 undersample=16, use_inventory=False,
                                 inventory_border=2,
                                 **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        test = bp.longitude(planet)
        show_info(bp, 'planet longitude (deg)', test*DPR, **options)
        test = bp.longitude(planet, reference='iau')
        show_info(bp, 'planet longitude wrt IAU frame (deg)', test*DPR, **options)
        test = bp.longitude(planet, lon_type='centric')
        show_info(bp, 'planet longitude centric (deg)', test*DPR, **options)
        test = bp.longitude(planet, lon_type='graphic')
        show_info(bp, 'planet longitude graphic (deg)', test*DPR, **options)
        test = bp.longitude(planet, lon_type='squashed')
        show_info(bp, 'planet longitude squashed (deg)', test*DPR, **options)
        test = bp.longitude(planet, direction='east')
        show_info(bp, 'planet longitude eastward (deg)', test*DPR, **options)
        test = bp.longitude(planet, minimum=-180)
        show_info(bp, 'planet longitude with -180 minimum (deg)', test*DPR, **options)
        test = bp.longitude(planet, reference='iau', minimum=-180)
        show_info(bp, 'planet longitude wrt IAU frame with -180 minimum (deg)',
                  test*DPR, **options)
        test = bp.longitude(planet, reference='sun')
        show_info(bp, 'planet longitude wrt Sun (deg)', test*DPR, **options)
        test = bp.longitude(planet, reference='sha')
        show_info(bp, 'planet longitude wrt SHA (deg)', test*DPR, **options)
        test = bp.longitude(planet, reference='obs')
        show_info(bp, 'planet longitude wrt observer (deg)', test*DPR, **options)
        test = bp.longitude(planet, reference='oha')
        show_info(bp, 'planet longitude wrt OHA (deg)', test*DPR, **options)
        test = bp.sub_observer_longitude(planet, reference='iau')
        show_info(bp, 'planet sub-observer longitude wrt IAU (deg)', test*DPR, **options)
        test = bp.sub_observer_longitude(planet, reference='sun', minimum=-180)
        show_info(bp, 'planet sub-observer longitude wrt Sun (deg)', test*DPR, **options)
        test = bp.sub_observer_longitude(planet, reference='obs', minimum=-180)
        show_info(bp, 'planet sub-observer longitude wrt observer (deg)', test*DPR,
                  **options)
        test = bp.sub_solar_longitude(planet, reference='iau')
        show_info(bp, 'planet sub-solar longitude wrt IAU (deg)', test*DPR, **options)
        test = bp.sub_solar_longitude(planet, reference='obs', minimum=-180)
        show_info(bp, 'planet sub-solar longitude wrt observer (deg)', test*DPR,
                  **options)
        test = bp.sub_solar_longitude(planet, reference='sun', minimum=-180)
        show_info(bp, 'planet sub-solar longitude wrt Sun (deg)', test*DPR,
                  **options)

    if moon is not None:
        test = bp.longitude(moon)
        show_info(bp, 'moon longitude (deg)', test*DPR, **options)
        test = bp.sub_observer_longitude(moon)
        show_info(bp, 'moon sub-observer longitude (deg)', test*DPR, **options)
        test = bp.sub_solar_longitude(moon)
        show_info(bp, 'moon sub-solar longitude (deg)', test*DPR, **options)

#===============================================================================
def exercise_ring(bp,
                  planet=None, moon=None, ring=None,
                  undersample=16, use_inventory=False, inventory_border=2,
                  **options):
    """generic unit tests for spheroid.py"""

    if ring is not None:
        test = bp.sub_observer_longitude(ring, 'sun', minimum=-180)
        show_info(bp, 'Ring observer-sun longitude (deg)', test*DPR, **options)


#*******************************************************************************
class Test_Spheroid(unittest.TestCase):

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
