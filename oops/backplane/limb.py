################################################################################
# oops/backplanes/limb_backplanes.py: Limb altitude backplanes.
################################################################################

import numpy as np

from polymath               import Qube
from oops.backplane         import Backplane
from oops.surface.polarlimb import PolarLimb

# Backplane names that can be "nested", such that the array mask propagates
# forward to each new backplane array that refers to it.
LIMB_BACKPLANES = ('limb_altitude',)

#===============================================================================
def limb_altitude(self, event_key, zmin=None, zmax=None, scaled=False):
    """Elevation of a limb point above the body's surface.

    Input:
        event_key       key defining the limb surface event.
        zmin            lower limit on altitude; lower values are masked.
        zmax            upper limit on altitude.
        scaled          if True, zmin and zmax are in units of the maximum body radius.
    """

    event_key = Backplane.standardize_event_key(event_key, default='LIMB')

    if scaled:
        body = self.get_body_and_modifier(event_key[-1])[0]
        radius = body.surface.radii.max()
        if zmin is not None:
            zmin = zmin * radius
        if zmin is not None:
            zmax = zmax * radius

    key = ('limb_altitude', event_key, zmin, zmax)
    if key in self.backplanes:
        return self.get_backplane(key)

    default_key = ('limb_altitude', event_key, None, None)
    if default_key not in self.backplanes:
        self._fill_limb_intercepts(event_key)

    altitude = self.get_backplane(default_key)
    if zmin is None and zmax is None:
        return altitude

    new_mask = False
    if zmin is not None:
        new_mask = Qube.or_(new_mask, altitude < zmin)
    if zmax is not None:
        new_mask = Qube.or_(new_mask, altitude > zmax)

    if np.any(new_mask):
        altitude = altitude.remask_or(new_mask)

    return self.register_backplane(key, altitude)

#===============================================================================
def _fill_limb_intercepts(self, event_key):
    """Internal method to fill in the limb intercept geometry backplanes.

    Input:
        event_key       key defining the limb surface event.
    """

    # Validate the surface type
    surface = Backplane.get_surface(event_key[1])
    if surface.COORDINATE_TYPE != 'limb':
        raise ValueError('invalid coordinate type for limb geometry: '
                         + surface.COORDINATE_TYPE)

    # Get the ring intercept coordinates
    event = self.get_surface_event(event_key)

    # Register the default backplanes
    self.register_backplane(('longitude', event_key, 'iau', 'east', 0,
                             'squashed'), event.coord1)
    self.register_backplane(('latitude', event_key, 'squashed'), event.coord2)
    self.register_backplane(('limb_altitude', event_key, None, None),
                            event.coord3)

#===============================================================================
def limb_longitude(self, event_key, reference='iau', direction='west',
                                    minimum=0, lon_type='centric'):
    """Longitude at the limb surface intercept point in the image.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        limb_altitude backplane key, in which case this
                        backplane inherits the mask of the given backplane
                        array.
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

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, LIMB_BACKPLANES,
                                                     default='LIMB')

    key = ('limb_longitude', event_key, reference, direction, minimum, lon_type)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    # Use the default longitude method
    longitude = self.longitude(*key[1:])
    return self.register_backplane(key, longitude)

#===============================================================================
def limb_latitude(self, event_key, lat_type='centric'):
    """Latitude at the surface intercept point in the image.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        limb_altitude backplane key, in which case this
                        backplane inherits the mask of the given backplane
                        array.
        lat_type        defines the type of latitude measurement:
                        'centric'   for planetocentric;
                        'graphic'   for planetographic;
                        'squashed'  for an intermediate latitude type used
                                    internally.
    """

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, LIMB_BACKPLANES,
                                                     default='LIMB')

    key = ('limb_latitude', event_key, lat_type)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    # Use the default latitude method
    latitude = self.latitude(event_key, lat_type)
    return self.register_backplane(key, latitude)

#===============================================================================
def limb_clock_angle(self, event_key):
    """Angular location around the limb, measured clockwise from the projected
    north pole.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        limb_altitude backplane key, in which case this
                        backplane inherits the mask of the given backplane
                        array.
    """

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, LIMB_BACKPLANES,
                                                     default='LIMB')

    key = ('limb_clock_angle', event_key)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    # Make sure the limb event is defined
    default_key = ('limb_clock_angle', event_key)
    if default_key not in self.backplanes:
        self._fill_limb_intercepts(event_key)

    surface = Backplane.get_surface(event_key[1])
    event = self.get_surface_event(event_key)

    polar_surface = PolarLimb(surface.ground, limits=surface.limits)
    event = polar_surface.apply_coords_to_event(event, obs=self.obs_event,
                                                       axes=2,
                                                       derivs=self.ALL_DERIVS)

    return self.register_backplane(key, event.coord2)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
