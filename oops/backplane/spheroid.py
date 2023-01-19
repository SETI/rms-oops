################################################################################
# oops/backplanes/spheroid.py: Spheroid/Ellipsoid backplanes
################################################################################

from polymath       import Scalar
from oops.backplane import Backplane

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
        minimum         the smallest numeric value of longitude in degrees,
                        either 0 or -180.
        lon_type        defines the type of longitude measurement:
                        'centric'   for planetocentric;
                        'graphic'   for planetographic;
                        'squashed'  for an intermediate longitude type used
                                    internally.
                        Note that lon_type is irrelevant to Spheroids but
                        matters for Ellipsoids.
    """

    if reference not in ('iau', 'sun', 'sha', 'obs', 'oha'):
        raise ValueError('invalid longitude reference: ' + repr(reference))

    if direction not in ('east', 'west'):
        raise ValueError('invalid longitude direction: ' + repr(direction))

    if minimum not in (0, -180):
        raise ValueError('invalid longitude minimum: ' + repr(minimum))

    if lon_type not in ('centric', 'graphic', 'squashed'):
        raise ValueError('invalid longitude type: ' + repr(lon_type))

    # Look up under the desired reference
    event_key = self.standardize_event_key(event_key)
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
        longitude = self.backplanes[key_typed]
    else:
        lon_squashed = self.backplanes[key_default]
        surface = self.get_surface(event_key[1])

        if lon_type == 'centric':
            longitude = surface.lon_to_centric(lon_squashed)
            longitude = self.register_backplane(key_typed, longitude)
        else:
            longitude = surface.lon_to_graphic(lon_squashed)
            longitude = self.register_backplane(key_typed, longitude)

    # Define the longitude relative to the reference value
    if reference != 'iau':
        if reference in ('sun', 'sha'):
            ref_lon = self._sub_solar_longitude(event_key)
        else:
            ref_lon = self._sub_observer_longitude(event_key)

        if reference in ('sha', 'oha'):
            ref_lon = ref_lon - Scalar.PI

        longitude = longitude - ref_lon

    # Reverse if necessary
    if direction == 'west':
        longitude = -longitude

    # Re-define the minimum
    if minimum == 0:
        longitude = longitude % Scalar.TWOPI
    else:
        longitude = (longitude + Scalar.PI) % Scalar.TWOPI - Scalar.PI

    return self.register_backplane(key, longitude)

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

    if lat_type not in ('centric', 'graphic', 'squashed'):
        raise ValueError('invalid latitude type: ' + repr(lat_type))

    # Look up under the desired reference
    event_key = self.standardize_event_key(event_key)
    key0 = ('latitude', event_key)
    key = key0 + (lat_type,)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with default keys, fill in those backplanes
    key_default = key0 + ('squashed',)
    if key_default not in self.backplanes:
        self._fill_surface_intercepts(event_key)

    # Fill in the values for this key
    latitude = self.backplanes[key_default]
    if lat_type == 'squashed':
        return latitude

    surface = self.get_surface(event_key[1])

    # Fill in the requested lon_type if necessary
    lon_key = ('longitude', event_key, 'iau', 'east', 0, 'squashed')
    longitude = self.backplanes[lon_key]

    if lat_type == 'centric':
        latitude = surface.lat_to_centric(latitude, longitude)
    else:
        latitude = surface.lat_to_graphic(latitude, longitude)

    return self.register_backplane(key, latitude)

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

    if event.surface.COORDINATE_TYPE != 'spherical':
        raise ValueError('invalid coordinate type for spheroidal geometry: '
                         + event.surface.COORDINATE_TYPE)

    self.register_backplane(lon_key, event.coord1)
    self.register_backplane(lat_key, event.coord2)

#===============================================================================
def _sub_observer_longitude(self, event_key):
    """Gridless sub-observer longitude. Used internally."""

    gridless_key = self.gridless_event_key(event_key)
    key = ('_sub_observer_longitude', gridless_key)

    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event(gridless_key)
    dep_ap = event.dep_ap
    longitude = dep_ap.to_scalar(1).arctan2(dep_ap.to_scalar(0)) % Scalar.TWOPI

    return self.register_backplane(key, longitude)

#===============================================================================
def _sub_observer_latitude(self, event_key):
    """Gridless sub-observer latitude. Used internally."""

    gridless_key = self.gridless_event_key(event_key)
    key = ('_sub_observer_latitude', gridless_key)

    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event(gridless_key)
    dep_ap = event.dep_ap
    latitude = (dep_ap.to_scalar(2) / dep_ap.norm()).arcsin()

    return self.register_backplane(key, latitude)

#===============================================================================
def _sub_solar_longitude(self, event_key):
    """Gridless sub-solar longitude. Used internally."""

    gridless_key = self.gridless_event_key(event_key)
    key = ('_sub_solar_longitude', gridless_key)

    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event(gridless_key, arrivals=True)
    neg_arr_ap = event.neg_arr_ap
    longitude = neg_arr_ap.to_scalar(1).arctan2(neg_arr_ap.to_scalar(0)) \
                % Scalar.TWOPI

    return self.register_backplane(key, longitude)

#===============================================================================
def _sub_solar_latitude(self, event_key):
    """Gridless sub-solar latitude. Used internally."""

    gridless_key = self.gridless_event_key(event_key)
    key = ('_sub_solar_latitude', gridless_key)

    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event(gridless_key, arrivals=True)
    neg_arr_ap = event.neg_arr_ap
    latitude = (neg_arr_ap.to_scalar(2) / neg_arr_ap.norm()).arcsin()

    return self.register_backplane(key, latitude)

################################################################################
# Surface geometry, path intercept versions
#   sub_observer_longitude()
#   sub_solar_longitude()
#   sub_observer_latitude()
#   sub_solar_latitude()
################################################################################

def sub_observer_longitude(self, event_key, reference='iau', direction='west',
                                            minimum=0):
    """Gridless sub-observer longitude.

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

    gridless_key = self.gridless_event_key(event_key)

    key0 = ('sub_observer_longitude', gridless_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.backplanes[key]

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        longitude = self.backplanes[key_default]
    else:
        longitude = self._sub_observer_longitude(gridless_key)
        longitude = self.register_backplane(key_default, longitude)

    if key == key_default:
        return longitude

    longitude = self._sub_longitude(event_key, longitude, reference=reference,
                                    direction=direction, minimum=minimum)
    return self.register_backplane(key, longitude)

#===============================================================================
def sub_solar_longitude(self, event_key, reference='iau',
                                         direction='west', minimum=0):
    """Gridless sub-solar longitude.

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

    gridless_key = self.gridless_event_key(event_key)

    key0 = ('sub_solar_longitude', event_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.backplanes[key]

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        longitude = self.backplanes[key_default]
    else:
        longitude = self._sub_solar_longitude(event_key)
        longitude = self.register_backplane(key_default, longitude)

    if key == key_default:
        return longitude

    longitude = self._sub_longitude(event_key, longitude, reference=reference,
                                    direction=direction, minimum=minimum)
    return self.register_backplane(key, longitude)

#===============================================================================
def _sub_longitude(self, event_key, longitude, reference='iau',
                                    direction='west', minimum=0):
    """Sub-solar or sub-observer longitude."""

    if reference not in ('iau', 'sun', 'sha', 'obs', 'oha'):
        raise ValueError('invalid longitude reference: ' + repr(reference))

    if direction not in ('east', 'west'):
        raise ValueError('invalid longitude direction: ' + repr(direction))

    if minimum not in (0, -180):
        raise ValueError('invalid longitude minimum: ' + repr(minimum))

    # Define the longitude relative to the reference value
    event_key = self.standardize_event_key(event_key)
    if reference != 'iau':
        if reference in ('sun', 'sha'):
            ref_lon = self._sub_solar_longitude(event_key)
        else:
            ref_lon = self._sub_observer_longitude(event_key)

        if reference in ('sha', 'oha'):
            ref_lon = ref_lon - Scalar.PI

        longitude = longitude - ref_lon

    # Reverse if necessary
    if direction == 'west':
        longitude = -longitude

    # Re-define the minimum
    if minimum == 0:
        longitude = longitude % Scalar.TWOPI
    else:
        longitude = (longitude + Scalar.PI) % Scalar.TWOPI - Scalar.PI

    return longitude

#===============================================================================
def sub_observer_latitude(self, event_key, lat_type='centric'):
    """Gridless sub-observer latitude at the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        lat_type        "centric" for planetocentric latitude;
                        "graphic" for planetographic latitude.
    """

    if lat_type not in ('centric', 'graphic'):
        raise ValueError('invalid latitude type: ' + repr(lat_type))

    gridless_key = self.gridless_event_key(event_key)
    key = ('sub_observer_latitude', gridless_key, lat_type)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event(gridless_key)
    dep_ap = event.apparent_dep()

    if lat_type == 'graphic':
        dep_ap = dep_ap.element_mul(event.surface.unsquash_sq)

    latitude = (dep_ap.to_scalar(2) / dep_ap.norm()).arcsin()
    return self.register_backplane(key, latitude)

#===============================================================================
def sub_solar_latitude(self, event_key, lat_type='centric'):
    """Gridless sub-solar latitude at the center of the disk.

    Input:
        event_key       key defining the event on the body's path.
        lat_type        "centric" for planetocentric latitude;
                        "graphic" for planetographic latitude.
    """

    if lat_type not in ('centric', 'graphic'):
        raise ValueError('invalid latitude type: ' + repr(lat_type))

    gridless_key = self.gridless_event_key(event_key)
    key = ('sub_solar_latitude', gridless_key, lat_type)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_gridless_event(gridless_key, arrivals=True)
    neg_arr_ap = event.neg_arr_ap

    if lat_type == 'graphic':
        neg_arr_ap = neg_arr_ap.element_mul(event.surface.unsquash_sq)

    latitude = (neg_arr_ap.to_scalar(2) / neg_arr_ap.norm()).arcsin()
    return self.register_backplane(key, latitude)

#===============================================================================
def lambert_law(self, event_key):
    """Lambert law model cos(incidence_angle) for the surface.

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lambert_law', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    lambert_law = self.incidence_angle(event_key).cos()
    lambert_law = lambert_law.mask_where(lambert_law.vals <= 0., 0.)
    return self.register_backplane(key, lambert_law)

#===============================================================================
def minnaert_law(self, event_key, k, k2=None, clip=0.2):
    """Minnaert law model for the surface.

    Input:
        event_key       key defining the surface event.
        k               The Minnaert exponent (for cos(i)).
        k2              Optional second Minnaert exponent (for cos(e)).
                        Defaults to k-1.
        clip            lower limit on cos(e). Needed because otherwise the
                        Minnaert law diverges near the limb. Default 0.2.
    """

    event_key = self.standardize_event_key(event_key)

    if k2 is None:
        k2 = k - 1
    key = ('minnaert_law', event_key, k, k2, clip)

    if key in self.backplanes:
        return self.backplanes[key]

    mu0 = self.lambert_law(event_key)
    mu = self.emission_angle(event_key).cos().clip(clip, None)
    minnaert_law = (mu0 ** k) * (mu ** k2)
    return self.register_backplane(key, minnaert_law)

#===============================================================================
def lommel_seeliger_law(self, event_key):
    """Lommel-Seeliger law model for the surface.

    Returns mu0 / (mu + mu0)

    Input:
        event_key       key defining the surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('lommel_seeliger_law', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    mu0 = self.incidence_angle(event_key).cos()
    mu  = self.emission_angle(event_key).cos()
    lommel_seeliger_law = mu0 / (mu + mu0)
    lommel_seeliger_law = lommel_seeliger_law.mask_where(mu0 <= 0., 0.)
    return self.register_backplane(key, lommel_seeliger_law)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

from oops.backplane.gold_master import register_test_suite
from oops.constants import DPR

def spheroid_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.body_names + bpt.limb_names:

        # Longitude
        cos_lat = bp.latitude(name).cos().min(builtins=True)
        bpt.gmtest(bp.longitude(name, 'iau') * DPR,
                   name + ' longitude, IAU (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, 'obs') * DPR,
                   name + ' longitude wrt observer (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, reference='obs', minimum=-180) * DPR,
                   name + ' longitude wrt observer, minimum -180 (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, 'oha') * DPR,
                   name + ' longitude wrt OHA (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, 'sun') * DPR,
                   name + ' longitude wrt Sun (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, 'sha') * DPR,
                   name + ' longitude wrt SHA (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)
        bpt.gmtest(bp.longitude(name, direction='east') * DPR,
                   name + ' longitude eastward (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=1)

        # Latitude
        bpt.gmtest(bp.latitude(name, lat_type='centric') * DPR,
                   name + ' latitude, planetocentric (deg)',
                   limit=0.001, radius=1)
        bpt.gmtest(bp.latitude(name, lat_type='graphic') * DPR,
                   name + ' latitude, planetographic (deg)',
                   limit=0.001, radius=1)

    for name in bpt.body_names:

        # Sub-observer longitude and latitude
        cos_lat = bp.sub_observer_latitude(name).cos().mean(builtins=True)
        bpt.gmtest(bp.sub_observer_longitude(name, reference='iau') * DPR,
                   name + ' sub-observer longitude, IAU (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.gmtest(bp.sub_observer_longitude(name, reference='sun', minimum=-180) * DPR,
                   name + ' sub-observer longitude wrt Sun (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.compare(bp.sub_observer_longitude(name, reference='obs', minimum=-180) * DPR,
                    0.,
                    name + ' sub-observer longitude wrt observer (deg)',
                    method='mod360')

        bpt.gmtest(bp.sub_observer_latitude(name, lat_type='centric') * DPR,
                   name + ' sub-observer latitude, planetocentric (deg)',
                   limit=0.001)
        bpt.gmtest(bp.sub_observer_latitude(name, lat_type='graphic') * DPR,
                   name + ' sub-observer latitude, planetographic (deg)',
                   limit=0.001)

        # Sub-solar longitude and latitude
        cos_lat = bp.sub_solar_latitude(name).cos().mean(builtins=True)
        bpt.gmtest(bp.sub_solar_longitude(name, reference='iau') * DPR,
                   name + ' sub-solar longitude wrt IAU (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.gmtest(bp.sub_solar_longitude(name, reference='obs', minimum=-180) * DPR,
                   name + ' sub-solar longitude wrt observer (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.compare(bp.sub_solar_longitude(name, reference='sun', minimum=-180) * DPR,
                    0.,
                    name + ' sub-solar longitude wrt Sun (deg)',
                    method='mod360')

        bpt.gmtest(bp.sub_solar_latitude(name, lat_type='centric') * DPR,
                   name + ' sub-solar latitude, planetocentric (deg)',
                   limit=0.001)
        bpt.gmtest(bp.sub_solar_latitude(name, lat_type='graphic') * DPR,
                   name + ' sub-solar latitude, planetographic (deg)',
                   limit=0.001)

        # Surface laws
        bpt.gmtest(bp.lambert_law(name),
                   name + ' as a Lambert law',
                   limit=1.e-6, radius=1)
        bpt.gmtest(bp.minnaert_law(name, 0.5),
                   name + ' as a Minnaert law (k=0.7)',
                   limit=1.e-6, radius=1)
        bpt.gmtest(bp.lommel_seeliger_law(name),
                   name + ' as a Lommel-Seeliger law',
                   limit=1.e-6, radius=1)

    # Test of an empty backplane
    for (planet, name) in bpt.planet_moon_pairs:
        if planet != 'PLUTO':
            bpt.compare(bp.longitude('STYX') * DPR,
                        0.,
                        'Styx longitude (deg)')
            break   # no need to repeat this test!

register_test_suite('spheroid', spheroid_test_suite)

################################################################################
# UNIT TESTS
################################################################################
import unittest
from oops.constants import DPR
from oops.backplane.unittester_support import show_info

#===============================================================================
def exercise_limb_longitude(bp,
                            planet=None, moon=None, ring=None,
                            undersample=16, use_inventory=False,
                            inventory_border=2,
                            **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        key = ('limb_altitude', planet + ':limb', 0., None)
        test = bp.limb_longitude(key, 'iau')
        show_info(bp, 'Limb longitude wrt IAU (deg)', test*DPR, **options)
        test = bp.limb_longitude(key, 'obs')
        show_info(bp, 'Limb longitude wrt observer (deg)', test*DPR, **options)
        test = bp.limb_longitude(key, reference='obs', minimum=-180)
        show_info(bp, 'Limb longitude wrt observer, -180 (deg)', test*DPR, **options)
        test = bp.limb_longitude(key, 'oha')
        show_info(bp, 'Limb longitude wrt OHA (deg)', test*DPR, **options)
        test = bp.limb_longitude(key, 'sun')
        show_info(bp, 'Limb longitude wrt Sun (deg)', test*DPR, **options)
        test = bp.limb_longitude(key, 'sha')
        show_info(bp, 'Limb longitude wrt SHA (deg)', test*DPR, **options)

#===============================================================================
def exercise_limb_latitude(bp,
                           planet=None, moon=None, ring=None,
                           undersample=16, use_inventory=False,
                           inventory_border=2,
                           **options):
    """generic unit tests for spheroid.py"""

    if planet is not None:
        key = ('limb_altitude', planet + ':limb', 0., None)
        test = bp.limb_latitude(key, lat_type='centric')
        show_info(bp, 'Limb planetocentric latitude (deg)', test*DPR, **options)
        test = bp.limb_latitude(key, lat_type='squashed')
        show_info(bp, 'Limb squashed latitude (deg)', test*DPR, **options)
        test = bp.limb_latitude(key, lat_type='graphic')
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
