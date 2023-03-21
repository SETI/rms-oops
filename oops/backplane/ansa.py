################################################################################
# oops/backplanes/ansa_backplanes.py: Ansa backplanes
################################################################################

from polymath import Scalar, Pair

from oops.backplane import Backplane

# Backplane names that can be "nested", such that the array mask propagates
# forward to each new backplane array that refers to it.
ANSA_BACKPLANES = ('ansa_radius',)

def ansa_radius(self, event_key, radius_type='positive', rmax=None):
    """Radius of the ring ansa intercept point in the image.

    Input:
        event_key       key defining the ring surface event.
        radius_type     'right'    for radii increasing rightward when prograde
                                   rotation pole is 'up';
                        'left'     for the opposite of 'right';
                        'positive' for all radii using positive values.
        rmax            maximum absolute value of the radius in km, if any.
    """

    # Validate inputs
    if radius_type not in ('right', 'left', 'positive'):
        raise ValueError('invalid radius_type: ' + repr(radius_type))

    # Look up under the desired radius type and maximum
    event_key = self.standardize_event_key(event_key, default='ANSA')
    key = ('ansa_radius', event_key, radius_type, rmax)
    if key in self.backplanes:
        return self.get_backplane(key)

    # Make sure the default is available
    key_default = ('ansa_radius', event_key, 'right', None)
    if key_default not in self.backplanes:
        self._fill_ansa_intercepts(event_key)

    # Get the unmasked backplane array
    radius = self.get_backplane(key_default)

    # Make sure the selected radius_type is available
    if radius_type == 'left':
        radius = self.register_backplane(key[:3] + (None,), -radius)
    elif radius_type == 'positive':
        radius = self.register_backplane(key[:3] + (None,), radius.abs())

    # If rmax is None, we're done
    if rmax is None:
        return radius

    # Otherwise, apply the mask
    mask = (radius.vals > rmax) | (radius.vals < -rmax)
    radius = radius.remask_or(mask)
    return self.register_backplane(key, radius)

#===============================================================================
def ansa_altitude(self, event_key):
    """Elevation of the ring ansa intercept point in the image.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        ansa_radius backplane key, in which case this backplane
                        inherits the mask of the given backplane array.
    """

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, ANSA_BACKPLANES,
                                                     default='ANSA')

    key = ('ansa_altitude', event_key)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    self._fill_ansa_intercepts(event_key)
    return self.get_backplane(key)

#===============================================================================
def ansa_longitude(self, event_key, reference='node'):
    """Longitude of the ansa intercept point in the image.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        ansa_radius backplane key, in which case this backplane
                        inherits the mask of the given backplane array.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
    """

    if reference not in ('aries', 'node', 'obs', 'oha', 'sun', 'sha'):
        raise ValueError('invalid longitude reference: ' + repr(reference))

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, ANSA_BACKPLANES,
                                                     default='ANSA')

    key = ('ansa_longitude', event_key, reference)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    # If it is not found with reference J2000, fill in those backplanes
    key_node = ('ansa_longitude', event_key, 'node')
    if key_node not in self.backplanes:
        self._fill_ansa_longitudes(event_key)

    # Now apply the reference longitude
    if reference == 'node':
        return self.get_backplane(key)

    if reference == 'aries':
        ref_lon = self._aries_ring_longitude(event_key)
    elif reference == 'sun':
        ref_lon = self._sub_solar_longitude(event_key)
    elif reference == 'sha':
        ref_lon = self._sub_solar_longitude(event_key) - Scalar.PI
    elif reference == 'obs':
        ref_lon = self._sub_observer_longitude(event_key)
    elif reference == 'oha':
        ref_lon = self._sub_observer_longitude(event_key) - Scalar.PI

    longitude = (self.get_backplane(key_node) - ref_lon) % Scalar.TWOPI
    return self.register_backplane(key, longitude)

#===============================================================================
def _fill_ansa_intercepts(self, event_key):
    """Internal method to fill in the ansa intercept geometry backplanes.

    Input:
        radius_type     'right'    for radii increasing rightward when prograde
                                   rotation pole is 'up';
                        'left'     for the opposite of 'right';
                        'positive' for all radii using positive values.
        rmax            maximum absolute value of the radius in km; None to
                        allow it to be defined by the event_key.
    """

    # Validate the surface type
    surface = self.get_surface(event_key[1])
    if surface.COORDINATE_TYPE != 'cylindrical':
        raise ValueError('invalid coordinate type for ansa geometry: '
                         + surface.COORDINATE_TYPE)

    # Get the ansa intercept coordinates
    event = self.get_surface_event(event_key)

    # Register the default backplanes
    self.register_backplane(('ansa_radius', event_key, 'right', None),
                            event.coord1)
    self.register_backplane(('ansa_altitude', event_key),
                            event.coord2)

#===============================================================================
def _fill_ansa_longitudes(self, event_key):
    """Internal method to fill in the ansa intercept longitude backplane."""

    # Validate the surface type
    surface = self.get_surface(event_key[1])
    if surface.COORDINATE_TYPE != 'cylindrical':
        raise ValueError('invalid coordinate type for ansa geometry: '
                         + surface.COORDINATE_TYPE)

    # Get the ansa intercept event
    event = self.get_surface_event(event_key)

    # Get the longitude in the associated ring plane
    lon = event.surface.ringplane.coords_from_vector3(event.state, axes=2,
                                                      derivs=self.ALL_DERIVS)[1]
    self.register_backplane(('ansa_longitude', event_key, 'node'), lon)

#===============================================================================
def ansa_radial_resolution(self, event_key):
    """Projected radial resolution in km/pixel at the ring ansa intercept.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        ansa_radius backplane key, in which case this backplane
                        inherits the mask of the given backplane array.
    """

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, ANSA_BACKPLANES,
                                                     default='ANSA')

    key = ('ansa_radial_resolution', event_key)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, derivs=True)
    if event.surface.COORDINATE_TYPE != 'cylindrical':
        raise ValueError('invalid coordinate type for ansa geometry: '
                         + event.surface.COORDINATE_TYPE)

    radius = event.coord1
    dr_duv = radius.d_dlos.chain(self.dlos_duv)
    resolution = dr_duv.join_items(Pair).norm()

    return self.register_backplane(key, resolution)

#===============================================================================
def ansa_vertical_resolution(self, event_key):
    """Projected radial resolution in km/pixel at the ring ansa intercept.

    Input:
        event_key       key defining the limb surface event. Alternatively, a
                        ansa_radius backplane key, in which case this backplane
                        inherits the mask of the given backplane array.
    """

    (event_key,
     backplane_key) = self._event_and_backplane_keys(event_key, ANSA_BACKPLANES,
                                                     default='ANSA')

    key = ('ansa_vertical_resolution', event_key)
    if backplane_key:
        return self._remasked_backplane(key, backplane_key)

    # If this backplane array is already defined, return it
    if key in self.backplanes:
        return self.get_backplane(key)

    event = self.get_surface_event(event_key, derivs=True)
    if event.surface.COORDINATE_TYPE != 'cylindrical':
        raise ValueError('invalid coordinate type for ansa geometry: '
                         + event.surface.COORDINATE_TYPE)

    altitude = event.coord2
    dz_duv = altitude.d_dlos.chain(self.dlos_duv)
    resolution = dz_duv.join_items(Pair).norm()

    return self.register_backplane(key, resolution)

################################################################################

# Add these functions to the Backplane module
Backplane._define_backplane_names(globals().copy())

################################################################################
# GOLD MASTER TESTS
################################################################################

import numpy as np
from oops.backplane.gold_master import register_test_suite
from oops.constants import PI

def ansa_test_suite(bpt):

    bp = bpt.backplane
    for name in bpt.ansa_names:
        bpt.gmtest(bp.ansa_radius(name),
                   name + ' radius (km)',
                   limit=0.1, radius=1)
        bpt.gmtest(bp.ansa_altitude(name),
                   name + ' altitude (km)',
                   limit=0.1, radius=1)
        bpt.gmtest(bp.ansa_radial_resolution(name),
                   name + ' radial resolution (km)',
                   limit=0.003, radius=1.5)
        bpt.gmtest(bp.ansa_vertical_resolution(name),
                   name + ' vertical resolution (km)',
                   limit=0.003, radius=1.5)

        bpt.gmtest(bp.ansa_longitude(name, 'node'),
                   name + ' longitude wrt node (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.gmtest(bp.ansa_longitude(name, 'aries'),
                   name + ' longitude wrt Aries (deg)',
                   method='mod360', limit=0.001, radius=1)

        longitude = bp.ansa_longitude(name, 'obs')
        bpt.gmtest(longitude,
                   name + ' longitude wrt observer (deg)',
                   method='mod360', limit=0.001, radius=1)
        bpt.compare(longitude - bp.ansa_longitude(name, 'oha'),
                    PI,
                    name + ' longitude wrt observer minus wrt OHA (deg)',
                    method='mod360', limit=1.e-13)

        longitude = bp.ansa_longitude(name, 'sun')
        bpt.gmtest(longitude,
                   name + ' longitude wrt Sun (deg)',
                   method = 'mod360', limit = 0.001, radius = 1)
        bpt.compare(longitude - bp.ansa_longitude(name, 'sha'),
                    PI,
                    name + ' longitude wrt Sun minus wrt SHA (deg)',
                    method='mod360', limit=1.e-13)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.ansa_names:

### this assumes center_distance is gridless, but sometimes it is gridded
### and gives an error...
        (ulimit, vlimit) = bp.center_distance(name) * pixel_duv * 1.e-3

### this addresses the immediate problem, but I thik dist should have
### only a single value
#        dist = bp.center_distance(name)
#        ulimit = dist * pixel_duv[0] * 1.e-3
#        vlimit = dist * pixel_duv[1] * 1.e-3

        # ansa_radius
        rad = bp.ansa_radius(name)
        drad_duv = rad.d_dlos.chain(bp.dlos_duv)
        (drad_du, drad_dv) = drad_duv.extract_denoms()

        drad = bp_u1.ansa_radius(name) - bp_u0.ansa_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_du,
                    name + ' radius d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        drad = bp_v1.ansa_radius(name) - bp_v0.ansa_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_dv,
                    name + ' radius d/dv self-check (km/pix)',
                    limit=vlimit, radius=1)

        # ansa_altitude
        alt = bp.ansa_altitude(name)
        dalt_duv = alt.d_dlos.chain(bp.dlos_duv)
        (dalt_du, dalt_dv) = dalt_duv.extract_denoms()

        dalt = bp_u1.ansa_altitude(name) - bp_u0.ansa_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_du,
                    name + ' altitude d/du self-check (km/pix)',
                    limit=ulimit, radius=1)

        dalt = bp_v1.ansa_altitude(name) - bp_v0.ansa_altitude(name)
        bpt.compare(dalt.wod/bpt.duv, dalt_dv,
                    name + ' altitude d/dv self-check (km/pix)',
                    limit=vlimit, radius=1)

register_test_suite('ansa', ansa_test_suite)

################################################################################
