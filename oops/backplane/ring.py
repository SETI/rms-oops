################################################################################
# oops/backplanes/ring.py: Ring backplanes
################################################################################

import numpy as np
from polymath import Scalar, Pair

from oops.backplane import Backplane
from oops.body      import Body
from oops.frame     import Frame
from oops.constants import PI, TWOPI

#===============================================================================
def ring_radius(self, event_key, rmin=None, rmax=None, lock_limits=False):
    """Radius of the ring intercept point in the observation.

    Input:
        event_key       key defining the ring surface event.
        rmin            minimum radius in km; None to allow it to be defined by
                        the event_key.
        rmax            maximum radius in km; None to allow it to be defined by
                        the event_key.
        lock_limits     if True, the rmin and rmax values will be applied to the
                        default event, so that all backplanes generated from
                        this event_key will have the same limits. This option
                        can only be applied the first time this event_key is
                        used.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ring_radius', event_key, rmin, rmax)
    if key in self.backplanes:
        return self.backplanes[key]

    default_key = ('ring_radius', event_key, None, None)
    if default_key not in self.backplanes:
        self._fill_ring_intercepts(event_key, rmin, rmax, lock_limits)

    rad = self.backplanes[default_key]
    if rmin is None and rmax is None:
        return rad

    if rmin is not None:
        mask0 = (rad < rmin)
    else:
        mask0 = False

    if rmax is not None:
        mask1 = (rad > rmax)
    else:
        mask1 = False

    rad = rad.mask_where(mask0 | mask1)
    self.register_backplane(key, rad)

    return rad

#===============================================================================
def ring_longitude(self, event_key, reference='node', rmin=None, rmax=None,
                         lock_limits=False):
    """Longitude of the ring intercept point in the image.

    Input:
        event_key       key defining the ring surface event.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
        rmin            minimum radius in km; None to allow it to be defined by
                        the event_key.
        rmax            maximum radius in km; None to allow it to be defined by
                        the event_key.
        lock_limits     if True, the rmin and rmax values will be applied to the
                        default event, so that all backplanes generated from
                        this event_key will have the same limits. This option
                        can only be applied the first time this event_key is
                        used.
    """

    event_key = self.standardize_event_key(event_key)
    assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

    # Look up under the desired reference
    key = ('ring_longitude', event_key, reference, rmin, rmax)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with reference='node', fill in those backplanes
    default_key = key[:2] + ('node', None, None)
    if default_key not in self.backplanes:
        self._fill_ring_intercepts(event_key, rmin, rmax, lock_limits)

    # Now apply the reference longitude
    reflon_key = key[:3] + (None, None)
    if reference == 'node':
        lon = self.backplanes[default_key]
    else:
        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self._sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self._sub_solar_longitude(event_key) - PI
        elif reference == 'obs':
            ref_lon = self._sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self._sub_observer_longitude(event_key) - PI

        lon = (self.backplanes[default_key] - ref_lon) % TWOPI

        self.register_backplane(reflon_key, lon)

    # Apply the radial mask if necessary
    if rmin is None and rmax is None:
        return self.backplanes[key]

    mask = self.ring_radius(event_key, rmin, rmax).mask
    lon = lon.mask_where(mask)
    self.register_backplane(key, lon)

    return lon

#===============================================================================
def radial_mode(self, backplane_key, cycles, epoch, amp, peri0, speed,
                      a0=0., dperi_da=0., reference='node'):
    """Radius shift based on a particular ring mode.

    Input:
        backplane_key   key defining a ring_radius backplane, possibly with
                        other radial modes.
        cycles          the number of radial oscillations in 360 degrees of
                        longitude.
        epoch           the time (seconds TDB) at which the mode parameters
                        apply.
        amp             radial amplitude of the mode in km.
        peri0           a longitude (radians) at epoch where the mode is at its
                        radial minimum at semimajor axis a0. For cycles == 0, it
                        is the phase at epoch, where a phase of 0 corresponds to
                        the minimum ring radius, with every particle at
                        pericenter.
        speed           local pattern speed in radians per second, as scaled by
                        the number of cycles.
        a0              the reference semimajor axis, used for slopes
        dperi_da        the rate of change of pericenter with semimajor axis,
                        measured at semimajor axis a0 in radians/km.
        reference       the reference longitude used to describe the mode; same
                        options as for ring_longitude
    """

    key = ('radial_mode', backplane_key, cycles, epoch, amp, peri0, speed,
                          a0, dperi_da, reference)

    if key in self.backplanes:
        return self.backplanes[key]

    # Get the backplane with modes
    rad = self.evaluate(backplane_key)

    # Get longitude and ring event time, without modes
    ring_radius_key = backplane_key
    while ring_radius_key[0] == 'radial_mode':
        ring_radius_key = ring_radius_key[1]

    (backplane_type, event_key, rmin, rmax) = ring_radius_key
    assert backplane_type == 'ring_radius', \
        'radial modes only apply to ring_radius backplanes'

    a = self.ring_radius(event_key)
    lon = self.ring_longitude(event_key, reference)
    time = self.event_time(event_key)

    # Add the new mode
    peri = peri0 + dperi_da * (a - a0) + speed * (time - epoch)
    if cycles == 0:
        mode = rad + amp * peri.cos()
    else:
        mode = rad + amp * (cycles * (lon - peri)).cos()

    # Replace the mask if necessary
    if rmin is None:
        if rmax is None:
            mask = False
        else:
            mask = (mode.vals > rmax)
    else:
        if rmax is None:
            mask = (mode.vals < rmin)
        else:
            mask = (mode.vals < rmin) | (mode.vals > rmax)

    if mask is not False:
        mode = mode.mask_where(mask)

    self.register_backplane(key, mode)
    return self.backplanes[key]

#===============================================================================
def _aries_ring_longitude(self, event_key):
    """Longitude of First Point of Aries from the ring ascending node.

    Primarily used internally. Longitudes are measured in the eastward
    (prograde) direction.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('_aries_ring_longitude', event_key)

    if key not in self.backplanes:
        event = self.get_gridless_event(event_key)
        frame = Frame.as_primary_frame(event.frame)
        lon = (-frame.node_at_time(event.time)) % TWOPI

        self.register_gridless_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def ring_azimuth(self, event_key, reference='obs'):
    """Angle from a reference direction to the local radial direction.

    The angle is measured in the prograde direction from a reference direction
    to the local radial, as measured at the ring intercept point and projected
    into the ring plane. This value is 90 degrees at the left ansa and 270
    degrees at the right ansa."

    The reference direction can be 'obs' for the apparent departing direction of
    the photon, or 'sun' for the (negative) apparent direction of the arriving
    photon.

    Input:
        event_key       key defining the ring surface event.
        reference       'obs' or 'sun'; see discussion above.
    """

    event_key = self.standardize_event_key(event_key)
    assert reference in ('obs', 'sun')

    # Look up under the desired reference
    key = ('ring_azimuth', event_key, reference)
    if key in self.backplanes:
        return self.backplanes[key]

    # If not found, fill in the ring events if necessary
    if ('ring_radius', event_key, None, None) not in self.backplanes:
        self._fill_ring_intercepts(event_key)

    # reference = 'obs'
    if reference == 'obs':
        event = self.get_surface_event(event_key)
        ref = event.apparent_dep()

    # reference = 'sun'
    else:
        event = self.get_surface_event_with_arr(event_key)
        ref = -event.apparent_arr()

    ref_angle = ref.to_scalar(1).arctan2(ref.to_scalar(0))
    rad_angle = event.pos.to_scalar(1).arctan2(event.pos.to_scalar(0))
    az = (rad_angle - ref_angle) % TWOPI
    self.register_backplane(key, az)

    return self.backplanes[key]

#===============================================================================
def ring_elevation(self, event_key, reference='obs', signed=True):
    """Angle from the ring plane to the photon direction.

    Evaluated at the ring intercept point. The angle is positive on the side of
    the ring plane where rotation is prograde; negative on the opposite side.

    The reference direction can be 'obs' for the apparent departing direction of
    the photon, or 'sun' for the (negative) apparent direction of the arriving
    photon.

    Input:
        event_key       key defining the ring surface event.
        reference       'obs' or 'sun'; see discussion above.
        signed          True for elevations on the retrograde side of the rings
                        to be negative; False for all angles to be non-negative.
    """

    event_key = self.standardize_event_key(event_key)
    assert reference in ('obs', 'sun')

    # Look up under the desired reference
    key = ('ring_elevation', event_key, reference, signed)
    if key in self.backplanes:
        return self.backplanes[key]

    key0 = ('ring_elevation', event_key, reference, True)
    if key0 in self.backplanes:
        return self.backplanes[key0].abs()

    # If not found, fill in the ring events if necessary
    if ('ring_radius', event_key, None, None) not in self.backplanes:
        self._fill_ring_intercepts(event_key)

    # reference = 'obs'
    if reference == 'obs':
        event = self.get_surface_event(event_key)
        dir = event.apparent_dep()

    # reference = 'sun'
    else:
        event = self.get_surface_event_with_arr(event_key)
        dir = -event.apparent_arr()

    el = Scalar.HALFPI - event.perp.sep(dir)
    self.register_backplane(key0, el)

    if not signed:
        el = el.abs()
        self.register_backplane(key, el)

    return self.backplanes[key]

#===============================================================================
def _fill_ring_intercepts(self, event_key, rmin=None, rmax=None,
                                lock_limits=False):
    """Internal method to fill in the ring intercept geometry backplanes.

    Input:
        event_key       key defining the ring surface event.
        rmax            lower limit to the ring radius in km. Smaller radii are
                        masked. Note that radii inside the planet are always
                        masked.
        rmax            upper limit to the ring radius in km. Larger radii are
                        masked.
        lock_limits     if True, the limits will be applied to the default
                        event, so that all backplanes generated from this
                        event_key will share the same limit. This can only be
                        applied the first time this event_key is used.
    """

    # Don't allow lock_limits if the backplane was already generated
    if rmin is None and rmax is None:
        lock_limits = False

    if lock_limits and event_key in self.surface_events:
        raise ValueError('lock_limits option disallowed for pre-existing ' +
                         'ring event key ' + str(event_key))

    # Get the ring intercept coordinates
    event = self.get_surface_event(event_key)
    if event.surface.COORDINATE_TYPE != 'polar':
        raise ValueError('ring geometry requires a polar coordinate system')

    # Apply the minimum radius if available
    planet_radius = self.min_ring_radius.get(event_key, None)
    if planet_radius:
        radius = event.coord1
        self.apply_mask_to_event(event_key, radius < planet_radius)

    # Apply the limits to the backplane if necessary
    if lock_limits:

        radius = event.coord1
        mask = False
        if rmin is not None:
            mask = mask | (radius < rmin)
        if rmax is not None:
            mask = mask | (radius > rmax)

        self.apply_mask_to_event(event_key, mask)
        event = self.get_surface_event(event_key)

    # Register the default ring_radius and ring_longitude backplanes
    self.register_backplane(('ring_radius', event_key, None, None),
                            event.coord1)
    self.register_backplane(('ring_longitude', event_key, 'node',
                             None, None), event.coord2)

    # Apply a mask just to these backplanes if necessary
    if rmin is not None or rmax is not None:

        radius = event.coord1
        mask = False
        if rmin is not None:
            mask = mask | (radius < rmin)
        if rmax is not None:
            mask = mask | (radius > rmax)

        self.register_backplane(('ring_radius', event_key, rmin, rmax),
                                radius.mask_where(mask))
        self.register_backplane(('ring_longitude', event_key, 'node',
                                 rmin, rmax), event.coord2.mask_where(mask))

#===============================================================================
def ring_incidence_angle(self, event_key, pole='sunward'):
    """Incidence angle of the arriving photons at the local ring surface.

    By default, angles are measured from the sunward pole and should always be
    <= pi/2. However, calculations for values relative to the IAU-defined north
    pole and relative to the prograde pole are also supported.

    Input:
        event_key       key defining the ring surface event.
        pole            'sunward'   for the pole on the illuminated face;
                        'north'     for the pole on the IAU north face;
                        'prograde'  for the pole defined by the direction of
                                    positive angular momentum.
    """

    assert pole in {'sunward', 'north', 'prograde'}

    # The sunward pole uses the standard definition of incidence angle
    if pole == 'sunward':
        return self.incidence_angle(event_key)

    event_key = self.standardize_event_key(event_key)

    # Return the cached copy if it exists
    key = ('ring_incidence_angle', event_key, pole)
    if key in self.backplanes:
        return self.backplanes[key]

    # Derive the prograde incidence angle if necessary
    key_prograde = key[:-1] + ('prograde',)
    if key_prograde not in self.backplanes:

        # Un-flip incidence angles where necessary
        incidence = self.incidence_angle(event_key)
        flip = self.backplanes[('ring_flip', event_key)]
        incidence = Scalar.PI * flip + (1. - 2.*flip) * incidence
        self.register_backplane(key_prograde, incidence)

    if pole == 'prograde':
        return self.backplanes[key_prograde]

    # If the ring is prograde, 'north' and 'prograde' are the same
    body_name = event_key[0]
    if ':' in body_name:
        body_name = body_name[:body_name.index(':')]

    body = Body.lookup(body_name)
    if not body.ring_is_retrograde:
        return self.backplanes[key_prograde]

    # Otherwise, flip the incidence angles and return a new backplane
    incidence = Scalar.PI - self.backplanes[key_prograde]
    self.register_backplane(key, incidence)
    return self.backplanes[key]

#===============================================================================
def ring_emission_angle(self, event_key, pole='sunward'):
    """Emission angle of the departing photons at the local ring surface.

    By default, angles are measured from the sunward pole, so the emission angle
    should be < pi/2 on the sunlit side and > pi/2 on the dark side of the
    rings. However, calculations for values relative to the IAU-defined north
    pole and relative to the prograde pole are also supported.

    Input:
        event_key       key defining the ring surface event.
        pole            'sunward' for the ring pole on the illuminated face;
                        'north' for the pole on the IAU-defined north face;
                        'prograde' for the pole defined by the direction of
                            positive angular momentum.
    """

    assert pole in {'sunward', 'north', 'prograde'}

    # The sunward pole uses the standard definition of emission angle
    if pole == 'sunward':
        return self.emission_angle(event_key)

    event_key = self.standardize_event_key(event_key)

    # Return the cached copy if it exists
    key = ('ring_emission_angle', event_key, pole)
    if key in self.backplanes:
        return self.backplanes[key]

    # Derive the prograde emission angle if necessary
    key_prograde = key[:-1] + ('prograde',)
    if key_prograde not in self.backplanes:

        # Un-flip incidence angles where necessary
        emission = self.emission_angle(event_key)
        flip = self.backplanes[('ring_flip', event_key)]
        emission = Scalar.PI * flip + (1. - 2.*flip) * emission
        self.register_backplane(key_prograde, emission)

    if pole == 'prograde' :
        return self.backplanes[key_prograde]

    # If the ring is prograde, 'north' and 'prograde' are the same
    body_name = event_key[0]
    if ':' in body_name:
        body_name = body_name[:body_name.index(':')]

    body = Body.lookup(body_name)
    if not body.ring_is_retrograde:
        return self.backplanes[key_prograde]

    # Otherwise, flip the emission angles and return a new backplane
    emission = Scalar.PI - self.backplanes[key_prograde]
    self.register_backplane(key, emission)
    return self.backplanes[key]

#===============================================================================
def ring_sub_observer_longitude(self, event_key, reference='node'):
    """Sub-observer longitude in the ring plane.

    Input:
        event_key       key defining the event on the center of the ring's path.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
    """

    assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

    # Look up under the desired reference
    key0 = ('ring_sub_observer_longitude', event_key)
    key = key0 + (reference,)
    if key in self.backplanes:
        return self.backplanes[key]

    # Generate longitude values
    default_key = key0 + ('node',)
    if default_key not in self.backplanes:
        lon = self._sub_observer_longitude(event_key)
        self.register_gridless_backplane(default_key, lon)

    # Now apply the reference longitude
    if reference != 'node':
        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self._sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self._sub_solar_longitude(event_key) - np.pi
        elif reference == 'obs':
            ref_lon = self._sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self._sub_observer_longitude(event_key) - np.pi

        lon = (self.backplanes[default_key] - ref_lon) % Scalar.TWOPI
        self.register_gridless_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def ring_sub_solar_longitude(self, event_key, reference='node'):
    """Sub-solar longitude in the ring plane.

    Input:
        event_key       key defining the event on the center of the ring's path.
        reference       defines the location of zero longitude.
                        'aries' for the First point of Aries;
                        'node'  for the J2000 ascending node;
                        'obs'   for the sub-observer longitude;
                        'sun'   for the sub-solar longitude;
                        'oha'   for the anti-observer longitude;
                        'sha'   for the anti-solar longitude, returning the
                                solar hour angle.
    """

    assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

    # Look up under the desired reference
    key0 = ('ring_sub_solar_longitude', event_key)
    key = key0 + (reference,)
    if key in self.backplanes:
        return self.backplanes[key]

    # If it is not found with reference='node', fill in those backplanes
    default_key = key0 + ('node',)
    if default_key not in self.backplanes:
        lon = self._sub_solar_longitude(event_key)
        self.register_gridless_backplane(default_key, lon)

    # Now apply the reference longitude
    if reference != 'node':
        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self._sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self._sub_solar_longitude(event_key) - np.pi
        elif reference == 'obs':
            ref_lon = self._sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self._sub_observer_longitude(event_key) - np.pi

        lon = (self.backplanes[default_key] - ref_lon) % Scalar.TWOPI

        self.register_gridless_backplane(key, lon)

    return self.backplanes[key]

#===============================================================================
def ring_center_incidence_angle(self, event_key, pole='sunward'):
    """Incidence angle of the arriving photons at the ring system center.

    By default, angles are measured from the sunward pole and should always be
    <= pi/2. However, calculations for values relative to the IAU-defined north
    pole and relative to the prograde pole are also supported.

    Input:
        event_key       key defining the ring surface event.
        pole            'sunward'   for the pole on the illuminated face;
                        'north'     for the pole on the IAU north face;
                        'prograde'  for the pole defined by the direction of
                                    positive angular momentum.
    """

    assert pole in {'sunward', 'north', 'prograde'}

    # The sunward pole uses the standard definition of incidence angle
    if pole == 'sunward':
        return self.center_incidence_angle(event_key)

    event_key = self.standardize_event_key(event_key)

    # Return the cached copy if it exists
    key0 = ('ring_center_incidence_angle', event_key)
    key = key0 + (pole,)
    if key in self.backplanes:
        return self.backplanes[key]

    # Derive the prograde incidence angle if necessary
    key_prograde = key0 + ('prograde',)
    if key_prograde in self.backplanes:
        incidence = self.backplanes[key_prograde]
    else:
        event = self.get_gridless_event_with_arr(event_key)

        # Sign on event.arr_ap is negative because photon is incoming
        latitude = (event.neg_arr_ap.to_scalar(2) /
                    event.arr_ap.norm()).arcsin()
        incidence = Scalar.HALFPI - latitude

        self.register_gridless_backplane(key_prograde, incidence)

    # If the ring is prograde, 'north' and 'prograde' are the same
    body_name = event_key[0]
    if ':' in body_name:
        body_name = body_name[:body_name.index(':')]

    body = Body.lookup(body_name)
    if not body.ring_is_retrograde:
        self.register_gridless_backplane(key, incidence)
        return incidence

    # Otherwise, flip the incidence angle and return a new backplane
    incidence = Scalar.PI - incidence
    self.register_gridless_backplane(key, incidence)

    return incidence

#===============================================================================
def ring_center_emission_angle(self, event_key, pole='sunward'):
    """Emission angle of departing photons at the center of the ring system.

    By default, angles are measured from the sunward pole, so the emission angle
    should be < pi/2 on the sunlit side and > pi/2 on the dark side of the
    rings. However, calculations for values relative to the IAU-defined north
    pole and relative to the prograde pole are also supported.

    Input:
        event_key       key defining the ring surface event.
        pole            'sunward' for the ring pole on the illuminated face;
                        'north' for the pole on the IAU-defined north face;
                        'prograde' for the pole defined by the direction of
                            positive angular momentum.
    """

    assert pole in {'sunward', 'north', 'prograde'}

    # The sunward pole uses the standard definition of emission angle
    if pole == 'sunward':
        return self.center_emission_angle(event_key)

    event_key = self.standardize_event_key(event_key)

    # Return the cached copy if it exists
    key0 = ('ring_center_emission_angle', event_key)
    key = key0 + (pole,)
    if key in self.backplanes:
        return self.backplanes[key]

    # Derive the prograde emission angle if necessary
    key_prograde = key0 + ('prograde',)
    if key_prograde in self.backplanes:
        emission = self.backplanes[key_prograde]
    else:
        event = self.get_gridless_event(event_key)

        latitude = (event.dep_ap.to_scalar(2) /
                    event.dep_ap.norm()).arcsin()
        emission = Scalar.HALFPI - latitude

        self.register_gridless_backplane(key_prograde, emission)

    # If the ring is prograde, 'north' and 'prograde' are the same
    body_name = event_key[0]
    if ':' in body_name:
        body_name = body_name[:body_name.index(':')]

    body = Body.lookup(body_name)
    if not body.ring_is_retrograde:
        self.register_gridless_backplane(key, emission)
        return emission

    # Otherwise, flip the emission angle and return a new backplane
    emission = Scalar.PI - emission
    self.register_gridless_backplane(key, emission)

    return emission

#===============================================================================
def ring_radial_resolution(self, event_key):
    """Projected radial resolution in km/pixel at the ring intercept point.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ring_radial_resolution', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event_w_derivs(event_key)
    assert event.surface.COORDINATE_TYPE == 'polar'

    rad = event.coord1
    drad_duv = rad.d_dlos.chain(self.dlos_duv)
    res = drad_duv.join_items(Pair).norm()

    self.register_backplane(key, res)
    return self.backplanes[key]

#===============================================================================
def ring_angular_resolution(self, event_key):
    """Projected angular resolution in radians/pixel at the ring intercept.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('ring_angular_resolution', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event_w_derivs(event_key)
    assert event.surface.COORDINATE_TYPE == 'polar'

    lon = event.coord2
    dlon_duv = lon.d_dlos.chain(self.dlos_duv)
    res = dlon_duv.join_items(Pair).norm()

    self.register_backplane(key, res)
    return self.backplanes[key]

#===============================================================================
def ring_gradient_angle(self, event_key=()):
    """Direction of the radius gradient at each pixel in the image.

    The angle is measured from the U-axis toward the V-axis.

    Input:
        event_key       key defining the ring surface event.
    """

    event_key = self.standardize_event_key(event_key)
    key = ('radial_gradient_angle', event_key)
    if key in self.backplanes:
        return self.backplanes[key]

    event = self.get_surface_event_w_derivs(event_key)
    assert event.surface.COORDINATE_TYPE == 'polar'

    rad = event.coord1
    drad_duv = rad.d_dlos.chain(self.dlos_duv)
    (drad_du, drad_dv) = drad_duv.join_items(Pair).to_scalars()

    clock = drad_dv.arctan2(drad_du)
    self.register_backplane(key, clock)

    return self.backplanes[key]

#===============================================================================
def ring_shadow_radius(self, event_key, ring_body):
    """Radius in the ring plane that casts a shadow at each point on this body.
    """

    event_key = self.standardize_event_key(event_key)
    ring_body = self.standardize_event_key(ring_body)[0]

    key = ('ring_shadow_radius', event_key, ring_body)
    if key not in self.backplanes:
        _ = self.get_surface_event_with_arr(event_key)
        ring_event = self.get_surface_event((ring_body,) + event_key)
        radius = ring_event.coord1
        self.register_backplane(key, radius)

    return self.backplanes[key]

#===============================================================================
def ring_radius_in_front(self, event_key, ring_body):
    """Radius in the ring plane that obscures this body."""

    event_key = self.standardize_event_key(event_key)
    ring_body = self.standardize_event_key(ring_body)[0]

    key = ('ring_in_front_radius', event_key, ring_body)
    if key not in self.backplanes:
        radius = self.ring_radius(ring_body)
        radius.mask_where(~self.where_intercepted(event_key))
        self.register_backplane(key, radius)

    return self.backplanes[key]

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
def exercise_resolution(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for ring.py"""
    
    if ring != None:
        test = bp.ring_radial_resolution(ring)
        show_info('Ring radial resolution (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_angular_resolution(ring)
        show_info('Ring angular resolution (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        radii = bp.ring_radius(ring)
        show_info('Ring angular resolution (km)', test * radii,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_radial_modes(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for ring.py"""
    
    if ring != None:
        test = bp.ring_radius(ring)
        show_info('Ring radius (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test0 = bp.ring_radius(ring, 70.e3, 100.e3)
        show_info('Ring radius, 70-100 kkm (km)', test0,   
                                    printing=printing, saving=saving, dir=dir)

        test1 = bp.radial_mode(test0.key, 40, 0., 1000., 0., 0., 100.e3)
        show_info('Ring radius, 70-100 kkm, mode 1 (km)', test1,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.radial_mode(test1.key, 40, 0., -1000., 0., 0., 100.e3)
        show_info('Ring radius, 70-100 kkm, mode 1 canceled (km)', test,   
                                    printing=printing, saving=saving, dir=dir)

        test2 = bp.radial_mode(test1.key, 25, 0., 500., 0., 0., 100.e3)
        show_info('Ring radius, 70-100 kkm, modes 1 and 2 (km)', test2,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_radius(ring).without_mask()
        show_info('Ring radius unmasked (km)', test,   
                                    printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_radial_longitude_azimuth(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for ring.py"""
    
    if ring != None:
        test = bp.ring_longitude(ring, reference='node')
        show_info('Ring longitude wrt node (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, 'node', 70.e3, 100.e3)
        show_info('Ring longitude wrt node, 70-100 kkm (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, reference='aries')
        show_info('Ring longitude wrt Aries (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, reference='obs')
        show_info('Ring longitude wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_azimuth(ring, 'obs')
        show_info('Ring azimuth wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_longitude(ring, 'obs')
        diff = test - compare
        show_info('Ring azimuth minus longitude wrt observer (deg)', 
                          diff*DPR, printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, reference='oha')
        show_info('Ring longitude wrt OHA (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, reference='sun')
        show_info('Ring longitude wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_azimuth(ring, reference='sun')
        show_info('Ring azimuth wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_longitude(ring, 'sun')
        diff = test - compare
        show_info('Ring azimuth minus longitude wrt Sun (deg)', diff*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_longitude(ring, reference='sha')
        show_info('Ring longitude wrt SHA (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_observer_longitude(ring, 'node')
        show_info('Ring sub-observer longitude wrt node (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_observer_longitude(ring, 'aries')
        show_info('Ring sub-observer longitude wrt Aries (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_observer_longitude(ring, 'sun')
        show_info('Ring sub-observer longitude wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_observer_longitude(ring, 'obs')
        show_info('Ring sub-observer longitude wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_solar_longitude(ring, 'node')
        show_info('Ring sub-solar longitude wrt node (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_solar_longitude(ring, 'aries')
        show_info('Ring sub-solar longitude wrt Aries (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_solar_longitude(ring, 'sun')
        show_info('Ring sub-solar longitude wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_sub_solar_longitude(ring, 'obs')
        show_info('Ring sub-solar longitude wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if planet != None:
        test = bp.ring_azimuth(planet+':ring', 'obs')
        show_info('Ring azimuth wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_azimuth(planet+':ring', 'obs')
        show_info('Ring azimuth wrt observer, unmasked (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_longitude(planet+':ring', 'obs')
        diff = test - compare
        show_info('Ring azimuth minus longitude wrt observer, unmasked (deg)', 
                          diff*DPR, printing=printing, saving=saving, dir=dir)

        test = bp.ring_azimuth(planet+':ring', reference='sun')
        show_info('Ring azimuth wrt Sun, unmasked (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_longitude(planet+':ring', 'sun')
        diff = test - compare
        show_info('Ring azimuth minus longitude wrt Sun, unmasked (deg)', 
                          diff*DPR, printing=printing, saving=saving, dir=dir)



#===========================================================================
def exercise_photometry(bp, obs, printing, saving, dir, 
                        planet=None, moon=None, ring=None, 
                        undersample=16, use_inventory=False, inventory_border=2):
    """Gerneric unit tests for ring.py"""
    
    if ring != None:
        test = bp.ring_incidence_angle(ring, 'sunward')
        show_info('Ring incidence angle, sunward (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_incidence_angle(ring, 'north')
        show_info('Ring incidence angle, north (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_incidence_angle(ring, 'prograde')
        show_info('Ring incidence angle, prograde (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.incidence_angle(ring)
        show_info('Ring incidence angle via incidence() (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_elevation(ring, reference='sun')
        show_info('Ring elevation wrt Sun (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_incidence_angle(ring, 'north')
        diff = test + compare
        show_info('Ring elevation wrt Sun plus north incidence (deg)', 
                          diff*DPR, printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_incidence_angle(ring, 'sunward')
        show_info('Ring center incidence angle, sunward (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_incidence_angle(ring, 'north')
        show_info('Ring center incidence angle, north (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_incidence_angle(ring, 'prograde')
        show_info('Ring center incidence angle, prograde (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_emission_angle(ring, 'sunward')
        show_info('Ring emission angle, sunward (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_emission_angle(ring, 'north')
        show_info('Ring emission angle, north (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_emission_angle(ring, 'prograde')
        show_info('Ring emission angle, prograde (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.emission_angle(ring)
        show_info('Ring emission angle via emission() (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_elevation(ring, reference='obs')
        show_info('Ring elevation wrt observer (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_emission_angle(ring, 'north')
        diff = test + compare
        show_info('Ring elevation wrt observer plus north emission (deg)', 
                           diff*DPR, printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_emission_angle(ring, 'sunward')
        show_info('Ring center emission angle, sunward (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_emission_angle(ring, 'north')
        show_info('Ring center emission angle, north (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        test = bp.ring_center_emission_angle(ring, 'prograde')
        show_info('Ring center emission angle, prograde (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

    if planet != None:
        test = bp.ring_elevation(planet+':ring', reference='sun')
        show_info('Ring elevation wrt Sun, unmasked (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_incidence_angle(planet+':ring', 'north')
        diff = test + compare
        show_info(
            'Ring elevation wrt Sun plus north incidence, unmasked (deg)', 
                          diff*DPR, printing=printing, saving=saving, dir=dir)

        test = bp.ring_elevation(planet+':ring', reference='obs')
        show_info('Ring elevation wrt observer, unmasked (deg)', test*DPR,   
                                    printing=printing, saving=saving, dir=dir)

        compare = bp.ring_emission_angle(planet+':ring', 'north')
        diff = test + compare
        show_info(
            'Ring elevation wrt observer plus north emission, unmasked (deg)',   
                           diff*DPR, printing=printing, saving=saving, dir=dir)




#*******************************************************************************
class Test_Ring(unittest.TestCase):


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
