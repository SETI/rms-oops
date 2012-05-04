################################################################################
# oops_/backplane.py: Backplane class
#
# 3/14/12 Created (MRS)
# 3/24/12 MRS - Revised handling of keys into the backplane dictionary, added
#   support for ansa surfaces; an entirely masked backplane no longer causes
#   problems.
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

    def __init__(self, obs, meshgrid=None, extras=(), t=Scalar(0.5)):
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

        # The surface_events dictionary comes in two versions, one with
        # derivatives and one without. Each dictionary is keyed by a tuple of
        # strings, where each string is the name of a body from which photons
        # depart. Each event is defined by a call to Surface.photon_to_event()
        # to the next. The last event is the arrival at the observation, which
        # is implied so it does not appear in the key. For example, ("SATURN",)
        # is the key for an event defining the departures of photons from the
        # surface of Saturn to the observer. Shadow calculations require a
        # pair of steps; for example, ("saturn","saturn_main_rings") is the key
        # for the event of intercept points on Saturn that fall in the path of
        # the photons arriving at Saturn's rings from the Sun.
        #
        # Note that body names in event keys are converted to lower case.

        self.surface_events_w_derivs = {(): self.obs_event}
        self.surface_events = {(): self.obs_event.plain()}

        # The path_events dictionary holds photon departure events from paths.
        # All photons originate from the Sun so this name is implied. For
        # example, ("SATURN",) is the key for the event of a photon departing
        # the Sun such that it later arrives at the Saturn surface arrival
        # event.

        self.path_events = {}

        # The backplanes dictionary holds every backplane that has been
        # calculated. This includes boolean backplanes, aka masks. A backplane
        # is keyed by (name of backplane, event_key, optional additional
        # parameters). The name of the backplane is always the name of the
        # backplane method that generates this backplane. For example,
        # ("phase_angle", ("saturn",)) is the key for a backplane of phase angle
        # values at the Saturn intercept points.
        #
        # If the function that returns a backplane requires additional
        # parameters, those appear in the tuple after the event key in the same
        # order that they appear in the calling function. For example,
        # ("latitude", ("saturn",), "graphic") is the key for the backplane of
        # planetographic latitudes at Saturn.

        self.backplanes = {}

    ############################################################################
    # Event manipulations
    #
    # Event keys are tuples (body_id, body_id, ...) where each body_id is a
    # a step along the path of a photon. The path always begins at the Sun and
    # and ends at the observer.
    ############################################################################

    @staticmethod
    def standardize_event_key(event_key):
        """Repairs the given event key to make it suitable as an index into the
        event dictionary. A string gets turned into a tuple."""

        if type(event_key) == type(""):
            return (event_key,)
        elif type(event_key) == type(()):
            return event_key
        else:
            raise ValueError("illegal event key type: " + str(type(event_key)))

    @staticmethod
    def standardize_backplane_key(backplane_key):
        """Repairs the given backplane key to make it suitable as an index into
        the backplane dictionary. A string is turned into a tuple. If the
        argument is a backplane already, the key is extracted from it."""

        if type(backplane_key) == type(""):
            return (backplane_key,)
        elif type(backplane_key) == type(()):
            return backplane_key
        elif isinstance(backplane_key, Array):
            return backplane_key.key
        else:
            raise ValueError("illegal backplane key type: " +
                              str(type(backplane_key)))

    @staticmethod
    def get_surface(surface_id):
        """Interprets the string identifying the surface and returns the
        associated surface. The string is normally a registered body ID in
        lower case, but it can be modified to indicate an surface or other
        associated surface."""

        modifier = None
        if surface_id.endswith(":ansa"):
            modifier = "ansa"
            body_id = surface_id[:-5]

        elif surface_id.endswith(":ring"):
            modifier = "ring"
            body_id = surface_id[:-5]

        else:
            modifier = None
            body_id = surface_id

        body = registry.body_lookup(body_id.upper())

        if modifier is None:
            return body.surface

        if modifier == "ring":
            return surface_.RingPlane(body.path_id, body.ring_frame_id,
                                      gravity=body.gravity)

        if modifier == "ansa":
            if body.surface.COORDINATE_TYPE == "polar":     # if it's a ring
                return surface_.Ansa(body.path_id, body.frame_id)
            else:                                           # if it's a planet
                return surface_.Ansa(body.path_id, body.ring_frame_id)

    @staticmethod
    def get_path(path_id):
        """Interprets the string identifying the path and returns the
        associated path."""

        return registry.body_lookup(path_id.upper()).path

    def get_surface_event(self, event_key):
        """Returns the event of photons leaving the specified body surface and
        arriving at a destination event identified by its key."""

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.surface_events[event_key]
        except KeyError: pass

        # The Sun is always treated as a path, not a surface
        if event_key[0].upper() == "SUN":
            return self.get_path_event(event_key)

        # call indicates an Ansa surface
#         if event_key[0].upper() == "ANSA":
#             return self.get_ansa_surface_event(event_key)

        # Always calculate derivatives for the first step from the observer
        if len(event_key) == 1:
            ignore = self.get_surface_event_w_derivs(event_key)
            return self.surface_events[event_key]

        # Create the event and save it in the dictionary
        dest = self.get_surface_event(event_key[1:])
        surface = Backplane.get_surface(event_key[0])

        event = surface.photon_to_event(dest)

        # Save extra information in the event object
        event.insert_subfield("event_key", event_key)
        event.insert_subfield("surface", surface)

        self.surface_events[event_key] = event
        return event

    def get_surface_event_w_derivs(self, event_key):
        """Returns the event of photons leaving the specified body surface and
        arriving at a destination event identified by its key. This version
        ensures that the surface derivatives are also included."""

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.surface_events_w_derivs[event_key]
        except KeyError: pass

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        surface = Backplane.get_surface(event_key[0])

        event = surface.photon_to_event(dest, derivs=True)

        # Save extra information in the event object
        event.insert_subfield("event_key", event_key)
        event.insert_subfield("surface", surface)

        self.surface_events_w_derivs[event_key] = event

        event_wo_derivs = event.plain()
        event_wo_derivs.event_key = event_key
        self.surface_events[event_key] = event_wo_derivs

        # Also save into the dictionary keyed by the string name alone
        # Saves the bother of typing parentheses and a comma when it's not
        # really necessary
        if len(event_key) == 1:
            self.surface_events_w_derivs[event_key[0]] = event
            self.surface_events[event_key[0]] = event_wo_derivs

        return event

    def get_path_event(self, event_key):
        """Returns the event of photons leaving the specified path and arriving
        at a destination event identified by its key."""

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.path_events[event_key]
        except KeyError: pass

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        path = Backplane.get_path(event_key[0])

        event = path.photon_to_event(dest)
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

    def mask_as_scalar(self, mask):
        """Converts a mask represented by a single boolean into a boolean array.
        """

        if isinstance(mask, np.ndarray): return Scalar(mask)
        if mask:
            return Scalar(np.ones(self.meshgrid.shape, dtype="bool"))
        else:
            return Scalar(np.zeros(self.meshgrid.shape, dtype="bool"))

    def register_backplane(self, key, backplane):
        """Inserts this backplane into the dictionary."""

        if isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)

        # For reference, we add the key as an attribute of each backplane object
        backplane = backplane.plain()
        backplane.key = key

        self.backplanes[key] = backplane

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

        event_key = Backplane.standardize_event_key(event_key)
        extras = (aberration, subfield, frame)

        (ra, dec) = self.get_event_with_arr(event_key).ra_and_dec(*extras)
        self.register_backplane(("right_ascension", event_key) + extras, ra)
        self.register_backplane(("declination", event_key) + extras, dec)

    def right_ascension(self, event_key=(), aberration=False, subfield="arr",
                              frame="j2000"):
        """Right ascension of the arriving or departing photon, optionally
        allowing for stellar aberration and for frames other than J2000.
        """

        event_key = Backplane.standardize_event_key(event_key)
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

        event_key = Backplane.standardize_event_key(event_key)
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

        event_key = Backplane.standardize_event_key(event_key)
        assert subfield in {"dep", "arr"}

        key = ("range", event_key, subfield)
        if key not in self.backplanes.keys():
            self.register_backplane(key,
                            self.light_time(event_key, subfield) * constants.C)

        return self.backplanes[key]

    def light_time(self, event_key, subfield="dep"):
        """Time in seconds between a photon departure event and an arrival
        event."""

        event_key = Backplane.standardize_event_key(event_key)
        assert subfield in {"dep", "arr"}

        key = ("light_time", event_key, subfield)
        if key not in self.backplanes.keys():
            if subfield == "arr":
                event = self.get_event_with_arr(event_key)
            else:
                event = self.get_surface_event(event_key)

            self.register_backplane(key, abs(event.subfields[subfield + "_lt"]))

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

        event_key = Backplane.standardize_event_key(event_key)
        key = ("incidence_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            self.register_backplane(key, event.incidence_angle())

        return self.backplanes[key]

    def emission_angle(self, event_key):
        """Emission angle of the departing photons at the local surface."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ("emission_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_surface_event(event_key)
            self.register_backplane(key, event.emission_angle())

        return self.backplanes[key]

    def phase_angle(self, event_key):
        """Phase angle between the arriving and departing photons.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("phase_angle", event_key)
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            self.register_backplane(key, event.phase_angle())

        return self.backplanes[key]

    def scattering_angle(self, event_key):
        """Scattering angle between the arriving and departing photons.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("scattering_angle", event_key)
        if key not in self.backplanes.keys():
            self.register_backplane(key, np.pi - self.phase_angle(event_key))

        return self.backplanes[key]

    ############################################################################
    # Ring plane geometry
    #
    # Radius keys are ("ring_radius", event_key)
    # Longitude keys are ("ring_longitude", event_key, reference)
    #   where reference can be "j2000", "obs", "oha", "sun" or "sha".
    ############################################################################

    def _fill_ring_intercepts(self, event_key):
        """Internal method to fill in the ring intercept geometry backplanes.
        """

        # Get the ring intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        #assert event.surface.COORDINATE_TYPE == "polar"

        (r,lon) = event.surface.event_as_coords(event, axes=2)

        self.register_backplane(("ring_radius", event_key), r)
        self.register_backplane(("ring_longitude", event_key, "j2000"), lon)

    def ring_radius(self, event_key):
        """Radius of the ring intercept point in the image.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ring_radius", event_key)
        if key not in self.backplanes.keys():
            self._fill_ring_intercepts(event_key)

        return self.backplanes[key]

    def ring_longitude(self, event_key, reference="j2000"):
        """Longitude of the ring intercept point in the image.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in {"j2000", "obs", "oha", "sun", "sha"}

        # Look up under the desired reference
        key0 = ("ring_longitude", event_key)
        key = key0 + (reference,)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If it is not found with reference J2000, fill in those backplanes
        key_j2000 = key0 + ("j2000",)
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

        lon = (self.backplanes[key0 + ("j2000",)] - ref_lon) % (2*np.pi)
        self.register_backplane(key0 + (reference,), lon)

        return self.backplanes[key]

    def _sub_observer_ring_longitude(self, event_key):
        """Sub-observer longitude at the ring center, but evaluated at the ring
        intercept time. Used only internally.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("_sub_observer_ring_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        obs_path = path_.Waypoint(self.obs_event.origin_id)
        ignore = obs_path.photon_from_event(center_event)

        surface = Backplane.get_surface(event_key[0])
        assert surface.COORDINATE_TYPE == "polar"
        (r,lon) = surface.coords_from_vector3(center_event.aberrated_dep(),
                                              axes=2)

        self.register_backplane(key, lon.unmasked())
        return self.backplanes[key]

    def _sub_solar_ring_longitude(self, event_key):
        """Sub-solar longitude at the ring center, but evaluated at the ring
        intercept time. Used only internally.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("_sub_solar_ring_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint("SUN").photon_to_event(center_event)

        surface = Backplane.get_surface(event_key[0])
        #assert event.surface.COORDINATE_TYPE == "polar"
        (r,lon) = surface.coords_from_vector3(-center_event.aberrated_arr(),
                                              axes=2)

        self.register_backplane(key, lon.unmasked())
        return self.backplanes[key]

    ############################################################################
    # Ring plane resolution
    #
    # Radius keys: ("ring_intercept_radial_resolution", event_key)
    # Longitude keys: ("ring_intercept_angular_resolution", event_key)
    ############################################################################

    def ring_radial_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring intercept point.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ring_radial_resolution", event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == "polar"

        (rad,lon) = event.surface.event_as_coords(event, axes=2,
                                                         derivs=(True,False))

        drad_duv = rad.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = drad_duv.as_pair().norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    def ring_angular_resolution(self, event_key):
        """Projected angular resolution in radians/pixel at the ring intercept
        point.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ring_angular_resolution", event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == "polar"

        (rad,lon) = event.surface.event_as_coords(event, axes=2,
                                                         derivs=(False,True))

        dlon_duv = lon.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = dlon_duv.as_pair().norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    ############################################################################
    # Ring ansa geometry
    #
    # Radius keys are ("ansa_radius", ring_event_key, radius_type)
    #   where radius_type = "right", "left" or "positive"
    # Elevation keys are ("ansa_elevation", ring_event_key)
    ############################################################################

    def _fill_ansa_intercepts(self, event_key):
        """Internal method to fill in the ansa intercept geometry backplanes.
        """

        # Get the ansa intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == "cylindrical"

        (r,z) = event.surface.event_as_coords(event, axes=2)

        self.register_backplane(("ansa_radius", event_key, "right"), r)
        self.register_backplane(("ansa_elevation", event_key), z)

    def ansa_radius(self, event_key, radius_type="right"):
        """Radius of the ring ansa intercept point in the image.
        """

        # Look up under the desired radius type
        event_key = Backplane.standardize_event_key(event_key)
        key0 = ("ansa_radius", event_key)
        key = key0 + (radius_type,)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If not found, look up the default "right"
        assert radius_type in {"right", "left", "positive"}

        key_default = key0 + ("right",)
        if key_default not in self.backplanes.keys():
            self._fill_ansa_intercepts(event_key)

            backplane = self.backplanes[key_default]
            if radius_type == "left":
                backplane = -backplane
            else:
                backplane = abs(backplane)

            self.register_backplane(key, backplane)

        return self.backplanes[key]

    def ansa_elevation(self, event_key):
        """Elevation of the ring ansa intercept point in the image.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ansa_elevation", event_key)
        if key not in self.backplanes.keys():
            self._fill_ansa_intercepts(event_key)

        return self.backplanes[key]

    def ansa_radial_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring ansa intercept
        point."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ansa_radial_resolution", event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == "cylindrical"

        (r,z) = event.surface.event_as_coords(event, axes=2,
                                                     derivs=(True,False))

        dr_duv = r.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = dr_duv.as_pair().norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    def ansa_vertical_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring ansa intercept
        point."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ("ansa_vertical_resolution", event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == "cylindrical"

        (r,z) = event.surface.event_as_coords(event, axes=2,
                                                     derivs=(False,True))

        dz_duv = z.d_dpos * event.pos.d_dlos * self.meshgrid.dlos_duv
        res = dz_duv.as_pair().norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    ############################################################################
    # Surface geometry
    #
    # Longitude keys are ("longitude", event_key, reference, direction, minimum)
    #   reference can be "iau", "sun", "sha", "obs" or "oha"
    #   where direction can be "east", "west"
    #   minimum can be 0 or -180
    #
    # Latitude keys are ("surface_intercept_latitude", event_key, lat_type)
    #   where lat_type can be "squashed", "centric" or "graphic"
    ############################################################################

    def _fill_surface_intercepts(self, event_key):
        """Internal method to fill in the surface intercept geometry backplanes.
        """

        # Get the surface intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == "spherical"

        (lon,lat) = event.surface.event_as_coords(event, axes=2, derivs=False)

        self.register_backplane(("longitude", event_key, "iau", "east", 0), lon)
        self.register_backplane(("latitude", event_key, "squashed"), lat)

    def longitude(self, event_key, reference="iau", direction="east",
                                                    minimum=0, ):
        """Longitude at the surface intercept point in the image.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in {"iau", "sun", "sha", "obs", "oha"}
        assert direction in {"east", "west"}
        assert minimum in {0, -180}

        # Look up under the desired reference
        key0 = ("longitude", event_key)
        key = key0 + (reference, direction, minimum)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # If it is not found with default keys, fill in those backplanes
        key_default = key0 + ("iau", "east", 0)
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

        self.register_backplane(key, lon)
        return self.backplanes[key]

    def latitude(self, event_key, lat_type="centric"):
        """Latitude at the surface intercept point in the image.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert lat_type in {"centric", "graphic", "squashed"}

        # Look up under the desired reference
        key0 = ("latitude", event_key)
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

        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == "spherical"

        if lat_type == "centric":
            lat = event.surface.lat_to_centric(lat)
        else:
            lat = event.surface.lat_to_graphic(lat)

        self.register_backplane(key, lat)
        return self.backplanes[key]

    def _sub_observer_surface_longitude(self, event_key):
        """Sub-observer longitude at the surface center, but evaluated at the
        surface intercept time. Used only internally.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("_sub_observer_surface_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        obs_path = path_.Waypoint(self.obs_event.origin_id)
        ignore = obs_path.photon_from_event(center_event)

        assert event.surface.COORDINATE_TYPE == "spherical"
        (lon,
        lat) = event.surface.coords_from_vector3(center_event.aberrated_dep(),
                                                 axes=2)

        self.register_backplane(key, lon.unmasked())
        return self.backplanes[key]

    def _sub_solar_surface_longitude(self, event_key):
        """Sub-solar longitude at the surface center, but evaluated at the
        surface intercept time. Used only internally.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("_sub_solar_surface_longitude", event_key)
        if key in self.backplanes.keys():
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (0,0,0), (0,0,0),
                                         event.origin_id, event.frame_id)
        ignore = path_.Waypoint("SUN").photon_to_event(center_event)

        assert event.surface.COORDINATE_TYPE == "spherical"
        (lon,
        lat) = event.surface.coords_from_vector3(-center_event.aberrated_arr(),
                                                 axes=2)

        self.register_backplane(key, lon.unmasked())
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

        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event_w_derivs(event_key)

        dpos_duv = event.pos.d_dlos * self.meshgrid.dlos_duv
        (minres, maxres) = surface_.Surface.resolution(dpos_duv)

        self.register_backplane(("finest_resolution", event_key), minres)
        self.register_backplane(("coarsest_resolution", event_key), maxres)

    def finest_resolution(self, event_key):
        """Projected spatial resolution in km/pixel in the optimal direction at
        the intercept point.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("finest_resolution", event_key)
        if key not in self.backplanes.keys():
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    def coarsest_resolution(self, event_key):
        """Projected spatial resolution in km/pixel in the worst direction at
        the intercept point.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ("coarsest_resolution", event_key)
        if key not in self.backplanes.keys():
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    ############################################################################
    # Masks
    ############################################################################

    def where_intercepted(self, event_key):
        """Returns a mask containing True wherever the surface was intercepted.
        """

        event_key  = Backplane.standardize_event_key(event_key)
        key = ("where_intercepted", event_key)
        if key not in self.backplanes.keys():
            event = self.get_surface_event(event_key)
            mask = self.mask_as_scalar(~event.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_inside_shadow(self, event_key, shadow_body):
        """Returns a mask containing True wherever the surface is in the shadow
        of a second body."""

        event_key = Backplane.standardize_event_key(event_key)
        shadow_body = Backplane.standardize_event_key(shadow_body)
        key = ("where_inside_shadow", event_key, shadow_body[0])
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            mask = self.mask_as_scalar(~event.mask & ~shadow_event.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_outside_shadow(self, event_key, shadow_body):
        """Returns a mask containing True wherever the surface is outside the
        shadow of a second body."""

        event_key  = Backplane.standardize_event_key(event_key)
        shadow_body = Backplane.standardize_event_key(shadow_body)
        key = ("where_outside_shadow", event_key, shadow_body[0])
        if key not in self.backplanes.keys():
            event = self.get_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            mask = self.mask_as_scalar(~event.mask & shadow_event.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_in_front(self, event_key, back_body):
        """Returns a mask containing True wherever the first surface is in front
        of the second surface."""

        event_key = Backplane.standardize_event_key(event_key)
        back_body  = Backplane.standardize_event_key(back_body)
        key = ("where_in_front", event_key, back_body[0])
        if key not in self.backplanes.keys():

            # A surface is in front if it is unmasked and the second surface is
            # either masked or further away.
            front_unmasked = ~self.get_surface_event(event_key).mask
            back_masked    =  self.get_surface_event(back_body).mask
            mask = self.mask_as_scalar(front_unmasked & (back_masked |
                    (self.range(event_key).vals < self.range(back_body).vals)))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_in_back(self, event_key, front_body):
        """Returns a mask containing True wherever first surface is behind the
        second surface."""

        event_key = Backplane.standardize_event_key(event_key)
        front_body = Backplane.standardize_event_key(front_body)

        key = ("where_in_back", event_key, front_body[0])
        if key not in self.backplanes.keys():

            # A surface is in back if it is unmasked and the second surface is
            # both unmasked and closer.
            back_unmasked  = ~self.get_surface_event(event_key).mask
            front_unmasked = ~self.get_surface_event(front_body).mask
            mask = self.mask_as_scalar(back_unmasked & front_unmasked &
                    (self.range(event_key).vals > self.range(front_body).vals))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_sunward(self, event_key):
        """Returns a mask containing True wherever the surface of a body is
        facing toward the Sun."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ("where_sunward",) + event_key
        if key not in self.backplanes.keys():
            incidence = self.incidence_angle(event_key)
            mask = self.mask_as_scalar((incidence.vals <= np.pi/2) &
                                       ~incidence.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_antisunward(self, event_key):
        """Returns a mask containing True wherever the surface of a body is
        facing away fron the Sun."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ("where_antisunward",) + event_key
        if key not in self.backplanes.keys():
            incidence = self.incidence_angle(event_key)
            mask = self.mask_as_scalar((incidence.vals > np.pi/2) &
                                       ~incidence.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    ############################################################################
    # Masks derived from backplanes
    ############################################################################

    def where_below(self, backplane_key, value):
        """Returns a mask that contains true wherever the value of the given
        backplane is less than or equal to the specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ("where_below", backplane_key, value)
        if key not in self.backplanes.keys():
            backplane = self.evaluate(backplane_key)
            mask = (backplane.vals <= value) & ~backplane.mask
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_above(self, backplane_key, value):
        """Returns a mask that contains true wherever the value of the given
        backplane is greater than or equal to the specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ("where_above", backplane_key, value)
        if key not in self.backplanes.keys():
            backplane = self.evaluate(backplane_key)
            mask = (backplane.vals >= value) & ~backplane.mask
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_between(self, backplane_key, low, high):
        """Returns a mask that contains true wherever the value of the given
        backplane is between the specified values, inclusive."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ("where_between", backplane_key, low, high)
        if key not in self.backplanes.keys():
            backplane = self.evaluate(backplane_key)
            mask = ((backplane.vals >= low) &
                    (backplane.vals <= high) & ~backplane.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    ############################################################################
    # Borders
    ############################################################################

    def _border_above_or_below(self, sign, backplane_key, value):
        """Defines the locus of points surrounding the region where the
        backplane either above (greater than or equal to) or below (less than or
        equal to) a specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)

        if sign > 0:
            key = ("border_above", backplane_key, value)
        else:
            key = ("border_below", backplane_key, value)

        if key not in self.backplanes.keys():
            backplane = sign * (self.evaluate(backplane_key) - value)
            border = np.zeros(self.meshgrid.shape, dtype="bool")

            axes = len(backplane.shape)
            for axis in range(axes):
                xbackplane = backplane.swapaxes(0, axis)
                xborder = border.swapaxes(0, axis)

                xborder[:-1] |= ((xbackplane[:-1].vals >= 0) &
                                 (xbackplane[1:].vals  < 0))
                xborder[1:]  |= ((xbackplane[1:].vals  >= 0) &
                                 (xbackplane[:-1].vals < 0))

            self.register_backplane(key, Scalar(border & ~backplane.mask))

        return self.backplanes[key]

    def border_above(self, backplane_key, value):
        """Defines the locus of points surrounding the region where the
        backplane is greater than or equal to a specified value."""

        return self._border_above_or_below(+1, backplane_key, value)

    def border_below(self, backplane_key, value):
        """Defines the locus of points surrounding the region where the
        backplane is less than or equal to a specified value."""

        return self._border_above_or_below(-1, backplane_key, value)

    def border_atop(self, backplane_key, value):
        """Defines the locus of points straddling the boder between the region
        where the backplane is below and where it is above the specified value.
        It selects the pixels that fall closest to the transition."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ("border_atop", backplane_key, value)
        if key not in self.backplanes.keys():
            absval = self.evaluate(backplane_key) - value
            sign = absval.sign()
            absval *= sign

            border = (absval == 0.)

            axes = len(absval.shape)
            for axis in range(axes):
                xabs = absval.vals.swapaxes(0, axis)
                xsign = sign.vals.swapaxes(0, axis)
                xborder = border.vals.swapaxes(0, axis)

                xborder[:-1] |= ((xsign[:-1] == -xsign[1:]) &
                                 (xabs[:-1] <= xabs[1:]))
                xborder[1:]  |= ((xsign[1:] == -xsign[:-1]) &
                                 (xabs[1:] <= xabs[:-1]))

            self.register_backplane(key, border)

        return self.backplanes[key]

    def _border_outside_or_inside(self, backplane_key, value):
        """Defines the locus of points that fall on the outer edge of a mask.
        "Outside" (value = False) identifies the first False pixels outside each
        area of True pixels; "Inside" (value = True) identifies the last True
        pixels adjacent to an area of False pixels."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)

        if value:
            key = ("border_inside", backplane_key)
        else:
            key = ("border_outside", backplane_key)

        if key not in self.backplanes.keys():
            backplane = self.evaluate(backplane_key) ^ (~value)
            # Reverses the backplane if value is False
            border = np.zeros(backplane.shape, dtype="bool")

            axes = len(backplane.shape)
            for axis in range(axes):
                xbackplane = backplane.swapaxes(0, axis)
                xborder = border.swapaxes(0, axis)

                xborder[:-1] |= ((xbackplane[:-1].vals ^ xbackplane[1:].vals) &
                                  xbackplane[:-1].vals)
                xborder[1:]  |= ((xbackplane[1:].vals ^ xbackplane[:-1].vals) &
                                  xbackplane[1:].vals)

            self.register_backplane(key, Scalar(border))

        return self.backplanes[key]

    def border_inside(self, backplane_key):
        """Defines the locus of True pixels adjacent to a region of False
        pixels."""

        return self._border_outside_or_inside(backplane_key, True)

    def border_outside(self, backplane_key):
        """Defines the locus of False pixels adjacent to a region of True
        pixels."""

        return self._border_outside_or_inside(backplane_key, False)

    ############################################################################
    # Method to access a backplane or mask by key
    ############################################################################

    # Here we use the class introspection capabilities of Python to provide a
    # general way to generate any backplane based on its key. This makes it
    # possible to access any backplane via its key rather than by making an
    # explicit call to the function that generates the key.

    # Here we keep track of all the function names that generate backplanes. For
    # security, we disallow evaluate() to access any function not in this list.

    CALLABLES = {
        "right_ascension", "declination",
        "range", "light_time",
        "incidence_angle", "emission_angle", "phase_angle", "scattering_angle",
        "ring_radius", "ring_longitude",
        "ring_radial_resolution", "ring_angular_resolution",
        "longitude", "latitude",
        "finest_resolution", "coarsest_resolution",
        "where_intercepted",
        "where_inside_shadow", "where_outside_shadow",
        "where_in_front", "where_in_back",
        "where_sunward", "where_antisunward",
        "where_below", "where_above", "where_between",
        "border_below", "border_above", "border_atop",
        "border_inside", "border_outside"}

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
UNITTEST_UNDERSAMPLE = 10

def show_info(title, array):
    """Internal method to print summary information and display images as
    desired."""

    global UNITTEST_PRINT
    if not UNITTEST_PRINT: return

    print title

    # Mask summary
    if array.vals.dtype == np.dtype("bool"):
        count = np.sum(array.vals)
        total = np.size(array.vals)
        percent = int(count / float(total) * 100. + 0.5)
        print "  ", (count, total-count),
        print (percent, 100-percent), "(True, False pixels)"

    # Unmasked backplane summary
    elif array.mask is False:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print "  ", minval
            else:
                print "  ", (minval, maxval), "(min, max)"

    # Masked backplane summary
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

        import oops_.registry as registry
        registry.initialize_frame_registry()
        registry.initialize_path_registry()
        registry.initialize_body_registry()

        import oops.inst.cassini.iss as iss

        if UNITTEST_LOGGING: config.LOGGING.on("        ")

        snap = iss.from_file(UNITTEST_FILESPEC)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)

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

        test = bp.longitude("saturn")
        show_info("Saturn longitude (deg)", test * constants.DPR)

        test = bp.longitude("saturn", direction="west")
        show_info("Saturn longitude westward (deg)", test * constants.DPR)

        test = bp.longitude("saturn", minimum=-180)
        show_info("Saturn longitude with -180 minimum (deg)",
                                                    test * constants.DPR)

        test = bp.longitude("saturn", reference="iau")
        show_info("Saturn longitude wrt IAU frame (deg)",
                                                    test * constants.DPR)

        test = bp.longitude("saturn", reference="sun")
        show_info("Saturn longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.longitude("saturn", reference="sha")
        show_info("Saturn longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.longitude("saturn", reference="obs")
        show_info("Saturn longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.longitude("saturn", reference="oha")
        show_info("Saturn longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.latitude("saturn", lat_type="centric")
        show_info("Saturn geocentric latitude (deg)", test * constants.DPR)

        test = bp.latitude("saturn", lat_type="graphic")
        show_info("Saturn geographic latitude (deg)", test * constants.DPR)

        test = bp.finest_resolution("saturn")
        show_info("Saturn finest surface resolution (km)", test)

        test = bp.coarsest_resolution("saturn")
        show_info("Saturn coarsest surface resolution (km)", test)

        test = bp.ring_radius("saturn_main_rings")
        show_info("Ring radius (km)", test)

        test = bp.ring_radius("saturn_main_rings").unmasked()
        show_info("Ring radius unmasked (km)", test)

        test = bp.ring_longitude("saturn_main_rings",reference="j2000")
        show_info("Ring longitude wrt J2000 (deg)", test * constants.DPR)

        test = bp.ring_longitude("saturn_main_rings", reference="sun")
        show_info("Ring longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.ring_longitude("saturn_main_rings", reference="sha")
        show_info("Ring longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.ring_longitude("saturn_main_rings", reference="obs")
        show_info("Ring longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.ring_longitude("saturn_main_rings", reference="oha")
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

        test = bp.ring_radial_resolution("saturn_main_rings")
        show_info("Ring radial resolution (km/pixel)", test)

        test = bp.ring_angular_resolution("saturn_main_rings")
        show_info("Ring angular resolution (deg/pixel)", test * constants.DPR)

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

        test = bp.evaluate(("where_antisunward", "saturn"))
        show_info("Saturn antisunward via evaluate()", test)

        test = bp.where_above(("incidence_angle", "saturn"), np.pi/2)
        show_info("Saturn antisunward via where_above()", test)

        test = bp.where_between(("incidence_angle", "saturn"), np.pi/2, 3.2)
        show_info("Saturn antisunward via where_between()", test)

        test = bp.where_intercepted("saturn")
        show_info("Saturn intercepted the old way", test)

        mask = bp.where_intercepted("saturn")
        test = bp.border_inside(mask)
        show_info("Saturn inside border", test)

        test = bp.border_outside(mask)
        show_info("Saturn outside border", test)

        test = bp.where_below(("ring_radius", "saturn_main_rings"), 100000.)
        show_info("Ring area below 100,000 km", test)

        test = bp.border_below(("ring_radius", "saturn_main_rings"), 100000.)
        show_info("Ring border below 100,000 km", test)

        test = bp.border_atop(("ring_radius", "saturn_main_rings"), 100000.)
        show_info("Ring border atop 100,000 km", test)

        test = bp.border_atop(("ring_radius", "saturn_main_rings"), 100000.)
        show_info("Ring border above 100,000 km", test)

        test = bp.evaluate(("border_atop", ("ring_radius",
                                            "saturn_main_rings"), 100000.))
        show_info("Ring border above 100,000 km via evaluate()", test)

        ########################
        # Testing ansa events and new notation

        test = bp.ring_radius("saturn_main_rings")
        show_info("Saturn main ring radius the old way (km)", test)

        test = bp.range("saturn:ring")
        show_info("Saturn:ring radius (km)", test)

        test = bp.range("saturn:ansa")
        show_info("Saturn:ansa range (km)", test)

        test = bp.ansa_radius("saturn:ansa")
        show_info("Saturn:ansa radius (km)", test)

        test = bp.ansa_elevation("saturn:ansa")
        show_info("Saturn:ansa elevation (km)", test)

        test = bp.ansa_radial_resolution("saturn:ansa")
        show_info("Saturn:ansa radial resolution (km)", test)

        test = bp.ansa_vertical_resolution("saturn:ansa")
        show_info("Saturn:ansa vertical resolution (km)", test)

        test = bp.ansa_vertical_resolution("saturn_main_rings:ansa")
        show_info("Saturn_main_ring:ansa vertical resolution (km)", test)

        test = bp.finest_resolution("saturn_main_rings:ansa")
        show_info("Saturn_main_ring:ansa finest resolution (km)", test)

        test = bp.finest_resolution("saturn_main_rings:ansa")
        show_info("Saturn_main_ring:ansa coarsest resolution (km)", test)

        ########################
        # Testing empty events
        # DO NOT PUT ANY MORE UNIT TESTS BELOW THIS LINE; BACKPLANE IS REPLACED

        snap = iss.from_file(UNITTEST_FILESPEC)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        bp = Backplane(snap, meshgrid)

        # Override some internals...
        old_obs = bp.obs_event
        bp.obs_event = Event(Scalar(old_obs.time, mask=True),
                             old_obs.pos, old_obs.vel,
                             old_obs.origin_id, old_obs.frame_id,
                             arr = old_obs.arr)

        bp.surface_events_w_derivs = {(): bp.obs_event}
        bp.surface_events = {(): bp.obs_event.plain()}

        test = bp.range("saturn")
        show_info("Range to Saturn, entirely masked (km)", test)

        test = bp.phase_angle("saturn")
        show_info("Phase angle at Saturn, entirely masked (deg)",
                                                test * constants.DPR)

        config.LOGGING.off()

        registry.initialize_frame_registry()
        registry.initialize_path_registry()
        registry.initialize_body_registry()
        iss.ISS.reset()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
