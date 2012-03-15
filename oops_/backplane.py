################################################################################
# oops_/backplane.py: Backplane class
#
# 3/14/12 Created (MRS)
################################################################################

import numpy as np
import oops_.config as config
import oops_.constants as constants
import oops_.registry as registry
import oops_.surface.all as surface_
import oops_.path.all as path_
from oops_.array.all import *
from oops_.event import Event
from oops_.meshgrid import Meshgrid

class Backplane(object):
    """Backplane is a class that supports the generation and manipulation of
    sets of backplanes associated with a particular observation. It caches
    intermediate results to speed up calculations."""

    def __init__(self, obs, meshgrid=None, extras=(), t=Scalar(0.)):
        """The constructor.

        Input:
            obs         the Observation object with which this backplane is
                        associated.
            meshgrid    the optional meshgrid defining the sampling of the FOV.
                        Default is to sample the center of every pixel.
            extras      any additional indices of the observation that might
                        affect the backplanes.
            t           a Scalar of fractional time offsets into each exposure;
                        default is 0.5. The shape of this Scalar will broadcast
                        with the shape of the meshgrid.
        """

        self.obs = obs

        if meshgrid is None:
            self.meshgrid = Meshgrid.for_fov(obs.fov, extras)
        else:
            self.meshgrid = meshgrid
        self.extras = extras

        self.obs_event = obs.event_at_grid(self.meshgrid, t)

        self.surface_events = {(): self.obs_event}
        self.path_events = {}
        self.backplanes = {}
        self.masks = {}

    ############################################################################
    # Event manipulations
    #
    # Event keys are tuples (body_id, body_id, ...) where each body_id is a
    # a step along the path of a photon. The path always begins at the Sun and
    # and ends at the observer.
    ############################################################################

    @staticmethod
    def fix_event_key(event_key):
        if type(event_key) == type(""):
            return (event_key,)
        elif type(event_key) == type(()):
            return event_key
        else:
            raise ValueError("illegal event key type: " + str(type(event_key)))

    def get_surface_event(self, event_key):
        """Returns the event of photons leaving the specified body surface and
        arriving at a destination event identified by its key."""

        event_key = Backplane.fix_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.surface_events[event_key]
        except KeyError: pass

        # The Sun is always treated as a path, not a surface
        if event_key[0].upper() == "SUN":
            return self.get_path_event(event_key)

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        surface = registry.body_lookup(event_key[0].upper()).surface

        # Always calculate derivatives for the first step from the observer
        derivs = (len(event_key) == 1)

        event = surface.photon_to_event(dest, derivs=derivs)
        event.event_key = event_key

        self.surface_events[event_key] = event
        if len(event_key) == 1:
            self.surface_events[event_key[0]] = event

        return event

    def get_path_event(self, event_key):
        """Returns the event of photons leaving the specified path and arriving
        at a destination event identified by its key."""

        event_key = Backplane.fix_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.path_events[event_key]
        except KeyError: pass

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        path = registry.body_lookup(event_key[0].upper()).path

        # Always calculate derivatives for the first step from the observer
        derivs = (len(event_key) == 1)

        event = path.photon_to_event(dest, derivs=derivs)
        event.event_key = event_key

        self.path_events[event_key] = event

        # For a tuple of length 1, also register under the name string
        if len(event_key) == 1:
            self.path_events[event_key[0]] = event

        return event

    def get_event_with_arr(self, event_key):
        """Returns the event object associated with the specified key, after
        ensuring that the arrival photons have been fille in."""

        event = self.get_surface_event(event_key)
        if event.arr == Empty():
            ignore = self.get_path_event(("sun",) + event_key)

        return event

    def mask_as_array(self, mask):
        """Converts a mask represented by a single boolean into a boolean array.
        """

        if isinstance(mask, np.ndarray): return mask
        if mask:
            return np.ones(self.meshgrid.shape, dtype="bool")
        else:
            return np.zeros(self.meshgrid.shape, dtyp="bool")

    ############################################################################
    # Sky geometry
    #
    # Right ascension keys are ("right_ascension", event_key, aberration,
    #                           subfield, frame)
    # Declination keys are ("declination", event_key, aberration,
    #                           subfield, frame)
    # where:
    #   aberration      False to leave the location of a planetary body fixed
    #                   with respect to the background stars;
    #                   True to shift the location based on the motion of the
    #                   observer relative to the SSB.
    #   subfield        "arr" to base the direction on an arriving photon;
    #                   "dep" to base the direction on a departing photon.
    #   frame           The name of the coordinate frame, typically "j2000" but
    #                   could be "B1950" for some purposes.
    ############################################################################

    def _fill_ra_dec(self, event_key, aberration=False, subfield="arr",
                                           frame="j2000"):
        """Internal method to fill in right ascension and declination
        backplanes."""

        event_key = Backplane.fix_event_key(event_key)
        extras = (aberration, subfield, frame)

        (ra, dec) = self.get_event_with_arr(event_key).ra_and_dec(*extras)
        self.backplanes[("right_ascension", event_key) + extras] = ra
        self.backplanes[("declination", event_key) + extras] = dec

    def right_ascension(self, event_key=(), aberration=False, subfield="arr",
                              frame="j2000"):
        """Right ascension of the arriving or departing photon, optionally
        allowing for stellar aberration and for frames other than J2000.
        """

        event_key = Backplane.fix_event_key(event_key)
        extras = (aberration, subfield, frame)
        key = ("right_ascension", event_key) + extras

        if key not in self.backplanes.keys():
            self._fill_ra_dec(event_key, aberration, subfield, frame)

        return self.backplanes[key]

    def declination(self, event_key=(), aberration=False, subfield="arr",
                          frame="j2000"):
        """Declination of the arriving or departing photon, optionally
        allowing for stellar aberration and for frames other than J2000.
        """

        event_key = Backplane.fix_event_key(event_key)
        extras = (aberration, subfield, frame)
        key = ("declination", event_key) + extras

        if key not in self.backplanes.keys():
            self._fill_ra_dec(event_key, aberration, subfield, frame)

        return self.backplanes[key]

    ############################################################################
    # Basic geometry
    #
    # Range keys are ("range", event_key, subfield)
    # Light time keys are ("light_time", event_key, subfield)
    #
    # Subfield is either "arr" for the range to the lighting source or "dep" for
    # the range to the observer.
    ############################################################################

    def range(self, event_key, subfield="dep"):
        """Range in km between a photon departure event to an arrival event."""

        event_key = Backplane.fix_event_key(event_key)
        assert subfield in {"dep", "arr"}

        key = ("range", event_key, subfield)
        if key not in self.backplanes.keys():
            self.backplanes[key] = self.light_time(event_key,
                                                   subfield) * constants.C

        return self.backplanes[key]

    def light_time(self, event_key, subfield="dep"):
        """Time in seconds between a photon departure event and an arrival
        event."""

        event_key = Backplane.fix_event_key(event_key)
        assert subfield in {"dep", "arr"}

        key = ("light_time", event_key, subfield)
        if key not in self.backplanes.keys():
            if subfield == "arr":
                event = self.get_event_with_arr(event_key)
            else:
                event = self.get_surface_event(event_key)

            self.backplanes[key] = abs(event.subfields[subfield + "_lt"])

        return self.backplanes[key]

    ############################################################################
    # Surface lighting geometry
    #
    # Indicence keys are ("incidence_angle", event_key)
    # Emission keys are ("emission_angle", event_key)
    # Phase angle keys are ("phase_angle", event_key)
    # Scattering angle keys are ("scattering_angle", event_key)
    ############################################################################

    def incidence_angle(self, event_key):
        """Incidence_angle angle of the arriving photons at the local surface.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("incidence_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            self.backplanes[key] = event.incidence_angle()

        return self.backplanes[key]

    def emission_angle(self, event_key):
        """Emission angle of the departing photons at the local surface."""

        event_key = Backplane.fix_event_key(event_key)
        key = ("emission_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_surface_event(event_key)
            self.backplanes[key] = event.emission_angle()

        return self.backplanes[key]

    def phase_angle(self, event_key):
        """Phase angle between the arriving and departing photons.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("phase_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            self.backplanes[key] = event.phase_angle()

        return self.backplanes[key]

    def scattering_angle(self, event_key):
        """Scattering angle between the arriving and departing photons.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("scattering_angle", event_key)
        if key not in self.backplanes.keys():
            self.backplanes[key] = np.pi - self.phase_angle(event_key)

        return self.backplanes[key]

    ############################################################################
    # Ring plane geometry
    #
    # Radius keys are ("ring_intercept_radius", event_key, mask)
    # Longitude keys are ("ring_intercept_longitude", event_key, mask,
    #                     reference)
    #   where reference can be "j2000", "obs", "oha", "sun" or "sha".
    ############################################################################

    def _fill_ring_intercepts(self, event_key):
        """Internal method to fill in the ring intercept geometry backplanes.
        """

        # Get the ring intercept coordinates
        event_key = Backplane.fix_event_key(event_key)
        event = self.get_surface_event(event_key)
        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, surface_.RingPlane)

        (r,lon) = surface.as_coords(event.pos, axes=2, derivs=True)

        # Save with mask
        self.backplanes[("ring_intercept_radius", event_key, True)] = r
        self.backplanes[("ring_intercept_longitude", event_key, True,
                                                                "j2000")] = lon

        # Also save without mask
        r = Scalar(r.vals, False)
        lon = Scalar(lon.vals, False)
        self.backplanes[("ring_intercept_radius", event_key, False)] = r
        self.backplanes[("ring_intercept_longitude", event_key, False,
                                                                "j2000")] = lon

    def ring_intercept_radius(self, event_key, masked=True):
        """Radius of the ring intercept point in the image.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("ring_intercept_radius", event_key, masked)
        if key not in self.backplanes.keys():
            self._fill_ring_intercepts(event_key)

        return self.backplanes[key]

    def ring_intercept_longitude(self, event_key, masked=True,
                                                  reference="j2000"):
        """Longitude of the ring intercept point in the image.
        """

        event_key = Backplane.fix_event_key(event_key)
        assert reference in {"j2000", "obs", "oha", "sun", "sha"}

        # Look up under the desired reference
        key0 = ("ring_intercept_longitude", event_key)
        key = key0 + (masked, reference)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If it is not found with reference J2000, fill in those backplanes
        key_j2000 = key0 + (masked, "j2000")
        if key_j2000 not in self.backplanes.keys():
            self._fill_ring_intercepts(event_key)

        # Now apply the reference longitude
        if reference == "j2000":
            return self.backplanes[key]

        if reference == "sun":
            ref_lon = self._sub_solar_ring_longitude(event_key)
        elif reference == "sha":
            ref_lon = self._sub_solar_ring_longitude(event_key) - np.pi
        elif reference == "obs":
            ref_lon = self._sub_observer_ring_longitude(event_key)
        elif reference == "oha":
            ref_lon = self._sub_observer_ring_longitude(event_key) - np.pi

        lon = (self.backplanes[key0 + (True, "j2000")] - ref_lon) % (2*np.pi)
        self.backplanes[key0 + (True, reference)] = lon
        self.backplanes[key0 + (False, reference)] = Scalar(lon.vals, False)

        return self.backplanes[key]

    def _sub_observer_ring_longitude(self, event_key):
        """Sub-observer longitude at the ring center, but evaluated at the ring
        intercept time. Used only internally.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("_sub_observer_ring_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint(self.obs.path_id).photon_from_event(
                                                                center_event)

        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, surface_.RingPlane)
        (r,lon) = surface.as_coords(center_event.aberrated_dep(), axes=2)

        self.backplanes[key] = Scalar(lon.vals)
        return self.backplanes[key]

    def _sub_solar_ring_longitude(self, event_key):
        """Sub-solar longitude at the ring center, but evaluated at the ring
        intercept time. Used only internally.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("_sub_solar_ring_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint("SUN").photon_to_event(center_event)

        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, surface_.RingPlane)
        (r,lon) = surface.as_coords(-center_event.aberrated_arr(), axes=2)

        self.backplanes[key] = Scalar(lon.vals)
        return self.backplanes[key]

    ############################################################################
    # Ring plane resolution
    #
    # Radius keys: ("ring_intercept_radial_resolution", event_key, masked)
    # Longitude keys: ("ring_intercept_angular_resolution", event_key, masked)
    ############################################################################

    def ring_intercept_radial_resolution(self, event_key, masked=True):
        """Projected radial resolution in km/pixel at the ring intercept point.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("ring_intercept_radial_resolution", event_key, masked)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event(event_key)
        rad = self.ring_intercept_radius(event_key, masked)
        drad_duv = rad.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = drad_duv.as_pair().norm()

        self.backplanes[key] = res
        return res

    def ring_intercept_angular_resolution(self, event_key, masked=True):
        """Projected angular resolution in radians/pixel at the ring intercept
        point.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("ring_intercept_angular_resolution", event_key, masked)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event(event_key)
        lon = self.ring_intercept_longitude(event_key, masked)
        dlon_duv = lon.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = dlon_duv.as_pair().norm()

        self.backplanes[key] = res
        return res

    ############################################################################
    # Surface geometry
    #
    # Longitude keys are ("surface_intercept_longitude", event_key,
    #                     direction, minimum, reference)
    #   where direction can be "east", "west"; minimum can be 0 or -180,
    #   reference can be "iau", "sun", "sha", "obs" or "oha".
    #
    # Latitude keys are ("surface_intercept_latitude", event_key, lat_type)
    #   where lat_type can be "squashed", "centric" or "graphic"
    ############################################################################

    def _fill_surface_intercepts(self, event_key):
        """Internal method to fill in the surface intercept geometry backplanes.
        """

        # Get the surface intercept coordinates
        event_key = Backplane.fix_event_key(event_key)
        event = self.get_surface_event(event_key)
        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, (surface_.Spheroid, surface_.Ellipsoid))

        (lon,lat) = surface.as_coords(event.pos, axes=2, derivs=False)

        self.backplanes[("surface_intercept_longitude", event_key, "east", 0, 
                                                        "iau")] = lon
        self.backplanes[("surface_intercept_latitude", event_key, 
                                                       "squashed")] = lat

    def surface_intercept_longitude(self, event_key, direction="east",
                                    minimum=0, reference="iau"):
        """Longitude at the surface intercept point in the image.
        """

        event_key = Backplane.fix_event_key(event_key)
        assert direction in {"east", "west"}
        assert minimum in {0, -180}
        assert reference in {"iau", "sun", "sha", "obs", "oha"}

        # Look up under the desired reference
        key0 = ("surface_intercept_longitude", event_key)
        key = key0 + (direction, minimum, reference)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If it is not found with default keys, fill in those backplanes
        key_default = key0 + ("east", 0, "iau")
        if key_default not in self.backplanes.keys():
            self._fill_surface_intercepts(event_key)

        # Fill in the values for this key
        if reference == "iau":
            ref_lon = 0.
        elif reference == "sun":
            ref_lon = self._sub_solar_surface_longitude(event_key)
        elif reference == "sha":
            ref_lon = self._sub_solar_surface_longitude(event_key) - np.pi
        elif reference == "obs":
            ref_lon = self._sub_observer_surface_longitude(event_key)
        elif reference == "oha":
            ref_lon = self._sub_observer_surface_longitude(event_key) - np.pi

        lon = self.backplanes[key_default] - ref_lon

        if direction == "west": lon = -lon

        if minimum == 0:
            lon %= (2.*np.pi)
        else:
            lon = (lon + np.pi) % (2.*np.pi) - np.pi

        self.backplanes[key] = lon
        return lon

    def surface_intercept_latitude(self, event_key, lat_type="centric"):
        """Latitude at the surface intercept point in the image.
        """

        event_key = Backplane.fix_event_key(event_key)
        assert lat_type in {"centric", "graphic", "squashed"}

        # Look up under the desired reference
        key0 = ("surface_intercept_latitude", event_key)
        key = key0 + (lat_type,)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If it is not found with default keys, fill in those backplanes
        key_default = key0 + ("squashed",)
        if key_default not in self.backplanes.keys():
            self._fill_surface_intercepts(event_key)

        # Fill in the values for this key
        lat = self.backplanes[key_default]
        if lat_type == "squashed":
            return lat

        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, (surface_.Spheroid, surface_.Ellipsoid))

        if lat_type == "centric":
            lat = surface.lat_to_centric(lat)
        else:
            lat = surface.lat_to_graphic(lat)

        self.backplanes[key] = lat
        return lat

    def _sub_observer_surface_longitude(self, event_key):
        """Sub-observer longitude at the surface center, but evaluated at the
        surface intercept time. Used only internally.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("_sub_observer_surface_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint(self.obs.path_id).photon_from_event(
                                                                center_event)

        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, (surface_.Spheroid, surface_.Ellipsoid))
        (lon,lat) = surface.as_coords(center_event.aberrated_dep(), axes=2)

        self.backplanes[key] = Scalar(lon.vals)
        return self.backplanes[key]

    def _sub_solar_surface_longitude(self, event_key):
        """Sub-solar longitude at the surface center, but evaluated at the
        surface intercept time. Used only internally.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("_sub_solar_surface_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint("SUN").photon_to_event(center_event)

        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, (surface_.Spheroid, surface_.Ellipsoid))
        (lon,lat) = surface.as_coords(-center_event.aberrated_arr(), axes=2)

        self.backplanes[key] = Scalar(lon.vals)
        return self.backplanes[key]

    ############################################################################
    # Surface resolution
    #
    # Radius keys: ("surface_intercept_finest_resolution", event_key)
    # Longitude keys: ("surface_intercept_coarsest_resolution", event_key)
    ############################################################################

    def _fill_surface_resolution(self, event_key):
        """Internal method to fill in the surface resolution backplanes.
        """

        event_key = Backplane.fix_event_key(event_key)
        event = self.get_surface_event(event_key)
        surface = registry.body_lookup(event_key[0].upper()).surface
        assert isinstance(surface, (surface_.Spheroid, surface_.Ellipsoid))

        dpos_duv = event.pos.d_dlos * self.meshgrid.dlos_duv
        (minres, maxres) = surface_.Surface.resolution(dpos_duv)

        self.backplanes["surface_intercept_finest_resolution",
                        event_key] = minres
        self.backplanes["surface_intercept_coarsest_resolution",
                        event_key] = maxres

    def surface_intercept_finest_resolution(self, event_key):
        """Projected spatial resolution in km/pixel in the optimal direction at
        the intercept point.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("surface_intercept_finest_resolution", event_key)
        if key not in self.backplanes.keys():
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    def surface_intercept_coarsest_resolution(self, event_key):
        """Projected spatial resolution in km/pixel in the worst direction at
        the intercept point.
        """

        event_key = Backplane.fix_event_key(event_key)
        key = ("surface_intercept_coarsest_resolution", event_key)
        if key not in self.backplanes.keys():
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    ############################################################################
    # Masks
    ############################################################################

    def where_intercepted(self, event_key):
        """Returns a mask containing True wherever the surface was intercepted.
        """

        event_key  = Backplane.fix_event_key(event_key)
        key = ("where_intercepted", event_key)
        if key not in self.masks.keys():
            event = self.get_surface_event(event_key)
            self.masks[key] = ~event.mask

        return self.masks[key]

    def where_inside_shadow(self, event_key, shadow_body):
        """Returns a mask containing True wherever the surface is in the shadow
        of a second body."""

        event_key = Backplane.fix_event_key(event_key)
        shadow_body = Backplane.fix_event_key(shadow_body)
        key = ("where_inside_shadow", event_key, shadow_body[0])
        if key not in self.masks.keys():
            event = self.get_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            self.masks[key] = self.mask_as_array(~event.mask &
                                                 ~shadow_event.mask)

        return self.masks[key]

    def where_outside_shadow(self, event_key, shadow_body):
        """Returns a mask containing True wherever the surface is outside the
        shadow of a second body."""

        event_key  = Backplane.fix_event_key(event_key)
        shadow_body = Backplane.fix_event_key(shadow_body)
        key = ("where_outside_shadow", event_key, shadow_body[0])
        if key not in self.masks.keys():
            event = self.get_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            self.masks[key] = self.mask_as_array(~event.mask &
                                                 shadow_event.mask)

        return self.masks[key]

    def where_in_front(self, event_key, back_body):
        """Returns a mask containing True wherever the first surface is in front
        of the second surface."""

        event_key = Backplane.fix_event_key(event_key)
        back_body  = Backplane.fix_event_key(back_body)
        key = ("where_in_front", event_key, back_body[0])
        if key not in self.masks.keys():

            # A surface is in front if it is unmasked and the second surface is
            # either masked or further away.
            front_unmasked = ~self.get_surface_event(event_key).mask
            back_masked    =  self.get_surface_event(back_body).mask
            self.masks[key] = (front_unmasked & (back_masked |
                    (self.range(event_key).vals < self.range(back_body).vals)))

        return self.masks[key]

    def where_in_back(self, event_key, front_body):
        """Returns a mask containing True wherever first surface is behind the
        second surface."""

        event_key = Backplane.fix_event_key(event_key)
        front_body = Backplane.fix_event_key(front_body)

        key = ("where_in_back", event_key, front_body[0])
        if key not in self.masks.keys():

            # A surface is in back if it is unmasked and the second surface is
            # both unmasked and closer.
            back_unmasked  = ~self.get_surface_event(event_key).mask
            front_unmasked = ~self.get_surface_event(front_body).mask
            self.masks[key] = (back_unmasked & front_unmasked &
                    (self.range(event_key).vals > self.range(front_body).vals))

        return self.masks[key]

    def where_sunward(self, event_key):
        """Returns a mask containing True wherever the surface of a body is
        facing toward the Sun."""

        event_key = Backplane.fix_event_key(event_key)
        key = ("where_sunward",) + event_key
        if key not in self.masks.keys():
            incidence = self.incidence_angle(event_key)
            self.masks[key] = (incidence.vals <= np.pi/2) & ~incidence.mask

        return self.masks[key]

    def where_antisunward(self, event_key):
        """Returns a mask containing True wherever the surface of a body is
        facing away fron the Sun."""

        event_key = Backplane.fix_event_key(event_key)
        key = ("where_antisunward",) + event_key
        if key not in self.masks.keys():
            incidence = self.incidence_angle(event_key)
            self.masks[key] = (incidence.vals > np.pi/2) & ~incidence.mask

        return self.masks[key]

    ############################################################################
    # Masks derived from backplanes
    ############################################################################

    def where_below(self, backplane_key, value):
        """Returns a mask that contains true wherever the value of the given
        backplane is less than or equal to the specified value."""

        plane = self.evaluate(backplane_key)
        return (plane.vals <= value) & ~plane.mask

    def where_above(self, backplane_key, value):
        """Returns a mask that contains true wherever the value of the given
        backplane is greater than or equal to the specified value."""

        plane = self.evaluate(key)
        return (plane.vals >= value) & ~plane.mask

    def where_between(self, backplane_key, low, high):
        """Returns a mask that contains true wherever the value of the given
        backplane is between the specified values, inclusive."""

        plane = self.evaluate(key)
        return (plane.vals >= low) & (plane.vals <= high) & ~plane.mask

    ############################################################################
    # Method to access a backplane or mask by key
    ############################################################################

    CALLABLES = {
        "right_ascension", "declination",
        "range", "light_time",
        "incidence_angle", "emission_angle", "phase_angle", "scattering_angle",
        "ring_intercept_radius", "ring_intercept_longitude",
        "ring_intercept_radial_resolution", "ring_intercept_angular_resolution",
        "surface_intercept_longitude", "surface_intercept_latitude",
        "surface_intercept_finest_resolution",
        "surface_intercept_coarsest_resolution",
        "where_intercepted",
        "where_inside_shadow", "where_outside_shadow",
        "where_in_front", "where_in_back",
        "where_sunward", "where_antisunward",
        "where_below", "where_above", "where_between"}

    def evaluate(self, backplane_key):
        """Evaluates the backplane or mask based on the given key. Equivalent
        to calling the function directly, but the name of the function is the
        first argument in the tuple passed to the function."""

        if type(backplane_key) == type(""): backplane_key = (backplane_key,)

        func = backplane_key[0]
        if func not in Backplane.CALLABLES:
            raise ValueError("unrecognized backplane function: " + func)

        return Backplane.__dict__[func].__call__(self, *backplane_key[1:])

################################################################################
# UNIT TESTS
################################################################################

import unittest

UNITTEST_PRINT = False
UNITTEST_LOGGING = False
UNITTEST_FILESPEC = "test_data/cassini/ISS/W1573721822_1.IMG"

def show_info(title, array):
    """Internal method to print summary information and display images as
    desired."""

    global UNITTEST_PRINT
    if not UNITTEST_PRINT: return

    print title

    if isinstance(array, np.ndarray):
        if array.dtype == np.dtype("bool"):
            count = np.sum(array)
            total = np.size(array)
            percent = int(count / float(total) * 100. + 0.5)
            print "  ", (count, total-count),
            print (percent, 100-percent), "(True, False pixels)"

        else:
            minval = np.min(array)
            maxval = np.max(array)
            if minval == maxval:
                print "  ", minval
            else:
                print "  ", (minval, maxval), "(min, max)"

    elif array.mask is False:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print "  ", minval
            else:
                print "  ", (minval, maxval), "(min, max)"

    else:
            print "  ", (np.min(array.vals),
                           np.max(array.vals)), "(unmasked min, max)"
            print "  ", (array.min(),
                           array.max()), "(masked min, max)"
            total = np.size(array.mask)
            masked = np.sum(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print "  ", (masked, total-masked),
            print         (percent, 100-percent),"(masked, unmasked pixels)"

####################################################
# TestCase begins here
####################################################

class Test_Backplane(unittest.TestCase):

    def runTest(self):

        import oops.inst.cassini.iss as iss

        if UNITTEST_LOGGING: config.LOGGING.on("        ")

        snap = iss.from_file(UNITTEST_FILESPEC)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=10, swap=True)

        bp = Backplane(snap, meshgrid)

        test = bp.right_ascension()
        show_info("Right ascension (deg)", test * constants.DPR)

        test = bp.right_ascension(aberration=True)
        show_info("Right ascension with aberration (deg)",
                                                    test * constants.DPR)

        test = bp.declination()
        show_info("Declination (deg)", test * constants.DPR)

        test = bp.declination(aberration=True)
        show_info("Declination with aberration (deg)",
                                                    test * constants.DPR)

        test = bp.range("saturn")
        show_info("Range to Saturn (km)", test)

        test = bp.range(("sun","saturn"))
        show_info("Saturn range to Sun (km)", test)

        test = bp.light_time("saturn")
        show_info("Light-time from Saturn (sec)", test)

        test = bp.incidence_angle("saturn")
        show_info("Saturn incidence angle (deg)", test * constants.DPR)

        test = bp.emission_angle("saturn")
        show_info("Saturn emission angle (deg)", test * constants.DPR)

        test = bp.phase_angle("saturn")
        show_info("Saturn phase angle (deg)", test * constants.DPR)

        test = bp.scattering_angle("saturn")
        show_info("Saturn scattering angle (deg)", test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn")
        show_info("Saturn longitude (deg)", test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", direction="west")
        show_info("Saturn longitude westward (deg)", test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", minimum=-180)
        show_info("Saturn longitude with -180 minimum (deg)",
                                                    test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", reference="iau")
        show_info("Saturn longitude wrt IAU frame (deg)",
                                                    test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", reference="sun")
        show_info("Saturn longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", reference="sha")
        show_info("Saturn longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", reference="obs")
        show_info("Saturn longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.surface_intercept_longitude("saturn", reference="oha")
        show_info("Saturn longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.surface_intercept_latitude("saturn", lat_type="centric")
        show_info("Saturn geocentric latitude (deg)", test * constants.DPR)

        test = bp.surface_intercept_latitude("saturn", lat_type="graphic")
        show_info("Saturn geographic latitude (deg)", test * constants.DPR)

        test = bp.surface_intercept_finest_resolution("saturn")
        show_info("Saturn finest surface resolution (km)", test)

        test = bp.surface_intercept_coarsest_resolution("saturn")
        show_info("Saturn coarsest surface resolution (km)", test)

        test = bp.ring_intercept_radius("saturn_main_rings", masked=True)
        show_info("Ring radius (km)", test)

        test = bp.ring_intercept_radius("saturn_main_rings", masked=False)
        show_info("Ring radius unmasked (km)", test)

        test = bp.ring_intercept_longitude("saturn_main_rings",
                                                    reference="j2000")
        show_info("Ring longitude wrt J2000 (deg)", test * constants.DPR)

        test = bp.ring_intercept_longitude("saturn_main_rings",
                                                    reference="sun")
        show_info("Ring longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.ring_intercept_longitude("saturn_main_rings",
                                                    reference="sha")
        show_info("Ring longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.ring_intercept_longitude("saturn_main_rings",
                                                    reference="obs")
        show_info("Ring longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.ring_intercept_longitude("saturn_main_rings",
                                                    reference="oha")
        show_info("Ring longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.range("saturn_main_rings")
        show_info("Range to rings (km)", test)

        test = bp.light_time("saturn_main_rings")
        show_info("Light time from rings (sec)", test)

        test = bp.range(("sun", "saturn_main_rings"))
        show_info("Ring range to Sun (km)", test)

        test = bp.incidence_angle("saturn_main_rings")
        show_info("Ring incidence angle (deg)", test * constants.DPR)

        test = bp.emission_angle("saturn_main_rings")
        show_info("Ring emission angle (deg)", test * constants.DPR)

        test = bp.phase_angle("saturn_main_rings")
        show_info("Ring phase angle (deg)", test * constants.DPR)

        test = bp.ring_intercept_radial_resolution("saturn_main_rings")
        show_info("Ring radial resolution (km/pixel)", test)

        test = bp.ring_intercept_angular_resolution("saturn_main_rings")
        show_info("Ring angular resolution (deg/pixel)",
                                                    test * constants.DPR)

        test = bp.where_intercepted("saturn")
        show_info("Saturn intercepted mask", test)

        test = bp.where_in_front("saturn", "saturn_main_rings")
        show_info("Saturn in front of rings", test)

        test = bp.where_in_back("saturn", "saturn_main_rings")
        show_info("Saturn in front of rings", test)

        test = bp.where_inside_shadow("saturn", "saturn_main_rings")
        show_info("Saturn in front of rings", test)

        test = bp.where_outside_shadow("saturn", "saturn_main_rings")
        show_info("Saturn in front of rings", test)

        test = bp.where_sunward("saturn")
        show_info("Saturn sunward", test)

        test = bp.where_antisunward("saturn")
        show_info("Saturn anti-sunward", test)

        test = bp.where_intercepted("saturn_main_rings")
        show_info("Rings intercepted mask", test)

        test = bp.where_in_front("saturn_main_rings", "saturn")
        show_info("Rings in front of Saturn", test)

        test = bp.where_in_back("saturn_main_rings", "saturn")
        show_info("Rings in front of Saturn", test)

        test = bp.where_inside_shadow("saturn_main_rings", "saturn")
        show_info("Rings in front of Saturn", test)

        test = bp.where_outside_shadow("saturn_main_rings", "saturn")
        show_info("Rings in front of Saturn", test)

        test = bp.where_sunward("saturn_main_rings")
        show_info("Rings sunward", test)

        test = bp.where_antisunward("saturn_main_rings")
        show_info("Rings anti-sunward", test)

        test = bp.right_ascension()
        show_info("Right ascension the old way (deg)", test * constants.DPR)

        test = bp.evaluate("right_ascension")
        show_info("Right ascension via evaluate() (deg)", test * constants.DPR)

        test = bp.where_intercepted("saturn")
        show_info("Saturn intercepted the old way", test)

        test = bp.evaluate(("where_intercepted", "saturn"))
        show_info("Saturn where intercepted via evaluate()", test)

        test = bp.where_sunward("saturn")
        show_info("Saturn sunward the old way", test)

        test = bp.evaluate(("where_sunward", "saturn"))
        show_info("Saturn sunward via evaluate()", test)

        test = bp.where_below(("incidence_angle", "saturn"), np.pi/2)
        show_info("Saturn sunward via where_below()", test)

        config.LOGGING.off()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
