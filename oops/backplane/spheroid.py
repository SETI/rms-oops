################################################################################
# oops/backplanes/spheroid.py: Spheroid/Ellipsoid backplanes
################################################################################

from polymath       import Scalar
from oops.backplane import Backplane
from oops.constants import C

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
    event_key = Backplane.standardize_event_key(event_key)
    key0 = ('longitude', event_key)
    key = key0 + (reference, direction, minimum, lon_type)
    if key in self.backplanes:
        return self.get_backplane(key)

    # If it is not found with default keys, fill in those backplanes
    # Note that longitudes default to eastward for right-handed
    # coordinates.
    key_default = key0 + ('iau', 'east', 0, 'squashed')
    if key_default not in self.backplanes:
        self._fill_surface_intercepts(event_key)

    # Fill in the required longitude type if necessary
    key_typed = key0 + ('iau', 'east', 0, lon_type)
    if key_typed in self.backplanes:
        longitude = self.get_backplane(key_typed)
    else:
        lon_squashed = self.get_backplane(key_default)
        surface = Backplane.get_surface(event_key[1])

        if lon_type == 'centric':
            longitude = surface.lon_to_centric(lon_squashed,
                                               derivs=self.ALL_DERIVS)
            longitude = self.register_backplane(key_typed, longitude)
        else:
            longitude = surface.lon_to_graphic(lon_squashed,
                                               derivs=self.ALL_DERIVS)
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
    event_key = Backplane.standardize_event_key(event_key)
    key0 = ('latitude', event_key)
    key = key0 + (lat_type,)
    if key in self.backplanes:
        return self.get_backplane(key)

    # If it is not found with default keys, fill in those backplanes
    key_default = key0 + ('squashed',)
    if key_default not in self.backplanes:
        self._fill_surface_intercepts(event_key)

    # Fill in the values for this key
    latitude = self.get_backplane(key_default)
    if lat_type == 'squashed':
        return latitude

    surface = Backplane.get_surface(event_key[1])

    # Fill in the requested lon_type if necessary
    lon_key = ('longitude', event_key, 'iau', 'east', 0, 'squashed')
    longitude = self.get_backplane(lon_key)

    if lat_type == 'centric':
        latitude = surface.lat_to_centric(latitude, longitude,
                                          derivs=self.ALL_DERIVS)
    else:
        latitude = surface.lat_to_graphic(latitude, longitude,
                                          derivs=self.ALL_DERIVS)

    return self.register_backplane(key, latitude)

#===============================================================================
def _fill_surface_intercepts(self, event_key):
    """Internal method to fill in the surface intercept geometry backplanes.
    """

    surface = Backplane.get_surface(event_key[1])

    # If this is actually a limb event, define the limb backplanes instead
    if surface.COORDINATE_TYPE == 'limb':
        self._fill_limb_intercepts(event_key)
        return

    # Validate the surface type
    if surface.COORDINATE_TYPE != 'spherical':
        raise ValueError('invalid coordinate type for spheroidal geometry: '
                         + surface.COORDINATE_TYPE)

    # Get the surface intercept coordinates
    event = self.get_surface_event(event_key)
    lon_key = ('longitude', event_key, 'iau', 'east', 0, 'squashed')
    lat_key = ('latitude', event_key, 'squashed')

    self.register_backplane(lon_key, event.coord1)
    self.register_backplane(lat_key, event.coord2)

#===============================================================================
def _sub_observer_longitude(self, event_key):
    """Gridless sub-observer longitude. Used internally."""

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('_sub_observer_longitude', gridless_key)

    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key)
    longitude = event.dep_ap.longitude(recursive=self.ALL_DERIVS)
        # Use the apparent departure direction seen at the body center
    return self.register_backplane(key, longitude)

#===============================================================================
def _sub_observer_latitude(self, event_key):
    """Gridless sub-observer latitude. Used internally."""

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('_sub_observer_latitude', gridless_key)

    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key)
    latitude = event.dep_ap.latitude(recursive=self.ALL_DERIVS)
        # Use the apparent departure direction seen at the body center
    return self.register_backplane(key, latitude)

#===============================================================================
def _sub_solar_longitude(self, event_key):
    """Gridless sub-solar longitude. Used internally."""

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('_sub_solar_longitude', gridless_key)

    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key, arrivals=True)
    longitude = event.neg_arr_ap.longitude(recursive=self.ALL_DERIVS)
        # Use the (negative) apparent arrival direction seen at the body center
    return self.register_backplane(key, longitude)

#===============================================================================
def _sub_solar_latitude(self, event_key):
    """Gridless sub-solar latitude. Used internally."""

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('_sub_solar_latitude', gridless_key)

    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key, arrivals=True)
    latitude = event.neg_arr_ap.latitude(recursive=self.ALL_DERIVS)
        # Use the (negative) apparent arrival direction seen at the body center
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

    gridless_key = Backplane.gridless_event_key(event_key)

    key0 = ('sub_observer_longitude', gridless_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.get_backplane(key)

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        longitude = self.get_backplane(key_default)
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

    gridless_key = Backplane.gridless_event_key(event_key)

    key0 = ('sub_solar_longitude', gridless_key)
    key = key0 + (reference, direction, minimum)
    if key in self.backplanes:
        return self.get_backplane(key)

    key_default = key0 + ('iau', 'east', 0)
    if key_default in self.backplanes:
        longitude = self.get_backplane(key_default)
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
    event_key = Backplane.standardize_event_key(event_key)
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

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('sub_observer_latitude', gridless_key, lat_type)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(gridless_key)
    dep_ap = event.dep_ap

    if lat_type == 'graphic':
        dep_ap = dep_ap.element_mul(event.surface.unsquash_sq,
                                    recursive=self.ALL_DERIVS)

    latitude = dep_ap.latitude(recursive=self.ALL_DERIVS)
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

    gridless_key = Backplane.gridless_event_key(event_key)
    key = ('sub_solar_latitude', gridless_key, lat_type)
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_gridless_event(gridless_key, arrivals=True)
    neg_arr_ap = event.neg_arr_ap

    if lat_type == 'graphic':
        neg_arr_ap = neg_arr_ap.element_mul(event.surface.unsquash_sq,
                                            recursive=self.ALL_DERIVS)

    latitude = neg_arr_ap.latitude(recursive=self.ALL_DERIVS)
    return self.register_backplane(key, latitude)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
