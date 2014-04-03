################################################################################
# oops/backplane.py: Backplane class
#
# TBD...
#   position(self, event_key, axis='x', apparent=True, frame="j2000")
#       returns an arbitrary position vector in an arbitrary frame
#   velocity(self, event_key, axis='x', apparent=True, frame="j2000")
#       returns an arbitrary velocity vector in an arbitrary frame
#   pole_clock_angle(self, event_key)
#       returns the projected angle of a body's pole vector on the sky
#   projected_radius(self, event_key, radius='long')
#       returns the projected angular radius of the body along its long or
#       short axis.
#   orbital_longitude(self, event_key, reference="j2000")
#       returns the orbital longitude of the body's path relative to the
#       barycenter of its motion.
################################################################################

import numpy as np
import os.path

from polymath import *

import oops.config    as config
import oops.constants as constants

from oops.surface_.surface   import Surface
from oops.surface_.ansa      import Ansa
#from oops.surface_.limb      import Limb
from oops.surface_.ringplane import RingPlane
from oops.path_.path         import Path, AliasPath
from oops.frame_.frame       import Frame
from oops.event              import Event
from oops.meshgrid           import Meshgrid
from oops.body               import Body

class Backplane(object):
    """Backplane is a class that supports the generation and manipulation of
    sets of backplanes associated with a particular observation. It caches
    intermediate results to speed up calculations."""

    def __init__(self, obs, meshgrid=None, time=None):
        """The constructor.

        Input:
            obs         the Observation object with which this backplane is
                        associated.
            meshgrid    the optional meshgrid defining the sampling of the FOV.
                        Default is to sample the center of every pixel.
            time        a Scalar of times. The shape of this Scalar will be
                        broadcasted with the shape of the meshgrid. Default is
                        to sample the midtime of every pixel.
        """

        self.obs = obs

        if meshgrid is None and obs.fov is not None:
            swap = obs.u_axis > obs.v_axis or obs.u_axis == -1
            self.meshgrid = Meshgrid.for_fov(obs.fov, swap=swap)
        else:
            self.meshgrid = meshgrid

        self.obs_event = obs.event_at_grid(self.meshgrid, time)
        self.obs_gridless_event = obs.gridless_event(self.meshgrid, time)

        self.shape = self.obs_event.shape

        # The surface_events dictionary comes in two versions, one with
        # derivatives and one without. Each dictionary is keyed by a tuple of
        # strings, where each string is the name of a body from which photons
        # depart. Each event is defined by a call to Surface.photon_to_event()
        # to the next. The last event is the arrival at the observation, which
        # is implied so it does not appear in the key. For example, ("SATURN",)
        # is the key for an event defining the departures of photons from the
        # surface of Saturn to the observer. Shadow calculations require a
        # pair of steps; for example, ('saturn','saturn_main_rings') is the key
        # for the event of intercept points on Saturn that fall in the path of
        # the photons arriving at Saturn's rings from the Sun.
        #
        # Note that body names in event keys are case-insensitive.

        self.surface_events_w_derivs = {(): self.obs_event}
        self.surface_events = {(): self.obs_event.without_derivs()}

        # The path_events dictionary holds photon departure events from paths.
        # All photons originate from the Sun so this name is implied. For
        # example, ("SATURN",) is the key for the event of a photon departing
        # the Sun such that it later arrives at the Saturn surface arrival
        # event.

        self.path_events = {}

        # The gridless_events dictionary keeps track of photon events at the
        # origin point of each defined surface. It uses the same keys as the
        # surface_events dictionary.

        self.gridless_events = {(): self.obs_gridless_event}

        # The backplanes dictionary holds every backplane that has been
        # calculated. This includes boolean backplanes, aka masks. A backplane
        # is keyed by (name of backplane, event_key, optional additional
        # parameters). The name of the backplane is always the name of the
        # backplane method that generates this backplane. For example,
        # ('phase_angle', ('saturn',)) is the key for a backplane of phase angle
        # values at the Saturn intercept points.
        #
        # If the function that returns a backplane requires additional
        # parameters, those appear in the tuple after the event key in the same
        # order that they appear in the calling function. For example,
        # ('latitude', ('saturn',), 'graphic') is the key for the backplane of
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
        """Repair an event key to make it suitable for indexing a dictionary.

        A string gets turned into a tuple.
        """

        if type(event_key) == type(""):
            return (event_key,)
        elif isinstance(event_key, basestring):
            return (str(event_key),)
        elif type(event_key) == type(()):
            return event_key
        else:
            raise ValueError("illegal event key type: " + str(type(event_key)))

    @staticmethod
    def standardize_backplane_key(backplane_key):
        """Repair a backplane key to make it suitable for indexing a dictionary.

        A string is turned into a tuple. Strings are converted to lower case. If
        the argument is a backplane already, the key is extracted from it.
        """

        if type(backplane_key) == type(""):
            return (backplane_key.lower(),)
        elif type(backplane_key) == type(()):
            return backplane_key
        elif isinstance(backplane_key, (Scalar,Boolean)):
            return backplane_key.key
        else:
            raise ValueError("illegal backplane key type: " +
                              str(type(backplane_key)))

    @staticmethod
    def get_surface(surface_id):
        """Return a surface based on its ID.

        The string is normally a registered body ID (case insensitive), but it
        can be modified with ":ansa", ":ring" or ":limb" to indicate an
        associated surface.
        """

        surface_id = surface_id.upper()

        modifier = None
        if surface_id.endswith(":ANSA"):
            modifier = "ANSA"
            body_id = surface_id[:-5]

        elif surface_id.endswith(":RING"):
            modifier = "RING"
            body_id = surface_id[:-5]

        elif surface_id.endswith(":LIMB"):
            modifier = "LIMB"
            body_id = surface_id[:-5]

        else:
            modifier = None
            body_id = surface_id

        body = Body.lookup(body_id)

        if modifier is None:
            return body.surface

        if modifier == "RING":
            return RingPlane(body.path, body.ring_frame, gravity=body.gravity)

        if modifier == "ANSA":
            if body.surface.COORDINATE_TYPE == 'polar':     # if it's a ring
                return Ansa.for_ringplane(body.surface)
            else:                                           # if it's a planet
                return Ansa(body.path, body.ring_frame)

#         if modifier == "LIMB":
#             return Limb(body.surface, limits=(0.,np.inf))

    @staticmethod
    def get_path(path_id):
        """Return a path based on its ID."""

        return Body.lookup(path_id.upper()).path

    def get_surface_event(self, event_key):
        """Return the photon departure event from a surface based on its key.
        """

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.surface_events[event_key]
        except KeyError: pass

        # The Sun is treated as a path, not a surface, unless it is listed first
        if event_key[0].upper() == 'SUN' and len(event_key) > 1:
            return self.get_path_event(event_key)

        # Look up the photon's departure surface and destination
        dest = self.get_surface_event(event_key[1:])
        surface = Backplane.get_surface(event_key[0])

        # Calculate derivatives for the first step from the observer, if allowed
        if len(event_key) == 1 and surface.intercept_DERIVS_ARE_IMPLEMENTED:
            try:
                event = self.get_surface_event_w_derivs(event_key)
                return self.surface_events[event_key]
            except NotImplementedError:
                pass

        # Create the event and save it in the dictionary
        event = surface.photon_to_event(dest)[0]

        # Save extra information in the event object
        event.event_key = event_key
        event.surface = surface

        self.surface_events[event_key] = event

        return event

    def get_surface_event_w_derivs(self, event_key):
        """The photon departure event from a surface including derivatives.
        """

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.surface_events_w_derivs[event_key]
        except KeyError: pass

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        surface = Backplane.get_surface(event_key[0])

        dest = dest.clone(subfields=True, recursive=True)
        dest.arr.insert_deriv('los', Vector3.IDENTITY, override=True)
        event = surface.photon_to_event(dest, derivs=True)[0]

        # Save extra information in the event object
        event.event_key = event_key
        event.surface = surface
        self.surface_events_w_derivs[event_key] = event

        # Make a copy without derivs, collapsing if possible
        event_wo_derivs = event.without_derivs().collapse_time()
        event_wo_derivs.event_key = event_key
        event_wo_derivs.surface = surface
        self.surface_events[event_key] = event_wo_derivs

        # Also save into the dictionary keyed by the string name alone
        # Saves the bother of typing parentheses and a comma when it's not
        # really necessary
        if len(event_key) == 1:
            self.surface_events_w_derivs[event_key[0]] = event
            self.surface_events[event_key[0]] = event_wo_derivs

        return event

    def get_path_event(self, event_key):
        """Return the departure event from a specified path."""

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.path_events[event_key]
        except KeyError: pass

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        path = Backplane.get_path(event_key[0])

        event = path.photon_to_event(dest)[0]
        event.event_key = event_key

        self.path_events[event_key] = event

        # For a tuple of length 1, also register under the name string
        if len(event_key) == 1:
            self.path_events[event_key[0]] = event

        return event

    def get_surface_event_with_arr(self, event_key):
        """Return the specified event with arrival photons filled in."""

        event = self.get_surface_event(event_key)
        if event.arr == Empty():
            new_event = AliasPath('SUN').photon_to_event(event)[1]
            new_event.event_key = event.event_key
            new_event.surface = event.surface
            self.surface_events[event_key] = new_event
            return new_event

        return event

    def get_gridless_event(self, event_key):
        """Return the gridless event identifying a photon departure from a path.
        """

        event_key = Backplane.standardize_event_key(event_key)

        # If the event already exists, return it
        try:
            return self.gridless_events[event_key]
        except KeyError: pass

        # Create the event and save it in the dictionary
        dest = self.get_gridless_event(event_key[1:])
        surface = Backplane.get_surface(event_key[0])

        path = Path.as_waypoint(surface.origin)
        event = path.photon_to_event(self.obs_gridless_event)[0]
        event = event.wrt_frame(surface.frame)

        # Save extra information in the event object
        event.event_key = event_key
        event.surface = surface

        self.gridless_events[event_key] = event
        return event

    def get_gridless_event_with_arr(self, event_key):
        """Return the gridless event with the arrival photons been filled in.
        """

        event_key = Backplane.standardize_event_key(event_key)

        event = self.get_gridless_event(event_key)
        if event.arr == Empty():
            new_event = AliasPath('SUN').photon_to_event(event)[1]
            new_event.event_key = event.event_key
            new_event.surface = event.surface
            self.gridless_events[event_key] = new_event
            return new_event

        return event

    def mask_as_boolean(self, mask):
        """Converts a mask represented by a single boolean into a Boolean.
        """

        if isinstance(mask, np.ndarray): return Boolean(mask)

        if mask:
            return Boolean(np.ones(self.shape, dtype='bool'))
        else:
            return Boolean(np.zeros(self.shape, dtype='bool'))

    def register_backplane(self, key, backplane):
        """Insert this backplane into the dictionary.

        If the backplane contains just a single value, it is expanded to the
        overall shape of the backplane."""

        if isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)

        # Under some circumstances a derived backplane can be a scalar
        if backplane.shape == () and self.shape != ():
            vals = np.empty(self.shape)
            vals[...] = backplane.vals
            backplane = Scalar(vals, backplane.mask)

        # For reference, we add the key as an attribute of each backplane object
        backplane = backplane.without_derivs()
        backplane.key = key

        self.backplanes[key] = backplane

    def register_gridless_backplane(self, key, backplane):
        """Insert this backplane into the dictionary.

        Same as register_backplane() but without the expansion of a scalar
        value."""

        if isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)
            backplane.key = backplane
        elif isinstance(backplane, (Scalar,Boolean)):
            backplane = backplane.without_derivs()
            backplane.key = backplane

        self.backplanes[key] = backplane

    ############################################################################
    # Sky geometry, surface intercept versions
    #   right_ascension()       right ascension of field of view, radians.
    #   declination()           declination of field of view, radians.
    ############################################################################

    def right_ascension(self, event_key=(), apparent=True, direction='arr'):
        """Right ascension of the arriving or departing photon

        Optionally, it allows for stellar aberration.

        Input:
            event_key       key defining the surface event, typically () to
                            refer to the observation.
            apparent        True to return the apparent direction of photons in
                            the frame of the event; False to return the purely
                            geometric directions of the photons.
            direction       'arr' to return the direction of an arriving photon;
                            'dep' to return the direction of a departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)

        key = ('right_ascension', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

    def declination(self, event_key=(), apparent=True, direction='arr'):
        """Declination of the arriving or departing photon.

        Optionally, it allows for stellar aberration.

        Input:
            event_key       key defining the surface event, typically () to
                            refer to the observation.
            apparent        True to return the apparent direction of photons in
                            the frame of the event; False to return the purely
                            geometric directions of the photons.
            direction       'arr' to base the direction on an arriving photon;
                            'dep' to base the direction on a departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)

        key = ('declination', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

    def _fill_ra_dec(self, event_key, apparent, direction):
        """Fill internal backplanes of RA and dec."""

        assert direction in ('arr', 'dep')

        event = self.get_surface_event_with_arr(event_key)
        (ra, dec) = event.ra_and_dec(apparent, direction)

        self.register_backplane(('right_ascension', event_key,
                                apparent, direction), ra)
        self.register_backplane(('declination', event_key,
                                apparent, direction), dec)

    def celestial_north_angle(self, event_key=()):
        """Direction of celestial north at each pixel in the image.

        The angle is measured from the U-axis toward the V-axis. This varies
        across the field of view due to spherical distortion and also any
        distortion in the FOV.

        Input:
            event_key       key defining the surface event, typically () to
                            refer to the observation.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('celestial_north_angle', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        temp_key = ('dlos_ddec', event_key)
        if not self.backplanes.has_key(temp_key):
            self._fill_dlos_dradec(event_key)

        dlos_ddec = self.backplanes[temp_key]
        duv_ddec = self.meshgrid.duv_dlos * dlos_ddec

        du_ddec_vals = duv_ddec.vals[...,0,0]
        dv_ddec_vals = duv_ddec.vals[...,1,0]
        clock = np.arctan2(dv_ddec_vals, du_ddec_vals)
        self.register_backplane(key, Scalar(clock, duv_ddec.mask))

        return self.backplanes[key]

    def celestial_east_angle(self, event_key=()):
        """Direction of celestial north at each pixel in the image.

        The angle is measured from the U-axis toward the V-axis. This varies
        across the field of view due to spherical distortion and also any
        distortion in the FOV.

        Input:
            event_key       key defining the surface event, typically () to
                            refer to the observation.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('celestial_east_angle', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        temp_key = ('dlos_dra', event_key)
        if not self.backplanes.has_key(temp_key):
            self._fill_dlos_dradec(event_key)

        dlos_dra = self.backplanes[temp_key]
        duv_dra = self.meshgrid.duv_dlos * dlos_dra

        du_dra_vals = duv_dra.vals[...,0,0]
        dv_dra_vals = duv_dra.vals[...,1,0]
        clock = np.arctan2(dv_dra_vals, du_dra_vals)
        self.register_backplane(key, Scalar(clock, duv_dra.mask))

        return self.backplanes[key]

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

        dlos_dra_vals = np.zeros(ra.shape + [3])
        dlos_dra_vals[...,0] = -sin_ra * cos_dec
        dlos_dra_vals[...,1] =  cos_ra * cos_dec

        dlos_j2000_dra = Vector3(dlos_dra_vals, ra.mask)

        dlos_ddec_vals = np.zeros(ra.shape + [3])
        dlos_ddec_vals[...,0] = -sin_dec * cos_ra
        dlos_ddec_vals[...,1] = -sin_dec * sin_ra
        dlos_ddec_vals[...,2] =  cos_dec

        dlos_j2000_ddec = Vector3(dlos_ddec_vals, ra.mask)

        # Rotate dlos from the J2000 frame to the image coordinate frame
        frame = self.obs.frame.wrt(Frame.J2000)
        xform = frame.transform_at_time(self.obs_event.time)

        dlos_dra  = xform.rotate(dlos_j2000_dra)
        dlos_ddec = xform.rotate(dlos_j2000_ddec)

        # Convert to a column matrix and save
        self.register_backplane(('dlos_dra',  event_key), dlos_dra.as_column())
        self.register_backplane(('dlos_ddec', event_key), dlos_ddec.as_column())

    ############################################################################
    # Sky geometry, path intercept versions
    #   center_right_ascension()    right ascension of path on sky, radians.
    #   center_declination()        declination of path on sky, radians.
    ############################################################################

    def center_right_ascension(self, event_key, apparent=True, direction='arr'):
        """Right ascension of the arriving or departing photon

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

        event_key = Backplane.standardize_event_key(event_key)

        key = ('center_right_ascension', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_center_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

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

        event_key = Backplane.standardize_event_key(event_key)

        key = ('center_declination', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_center_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

    def _fill_center_ra_dec(self, event_key, apparent, direction):
        """Internal method to fill in RA and dec for the center of a body."""

        assert direction in ('arr', 'dep')

        event = self.get_surface_event_with_arr(event_key)
        (ra, dec) = event.ra_and_dec(apparent, direction)

        self.register_gridless_backplane(
                ('center_right_ascension', event_key, apparent, direction), ra)
        self.register_gridless_backplane(
                ('center_declination', event_key, apparent, direction), dec)

    ############################################################################
    # Basic body geometry, surface intercept versions
    #   distance()          distance traveled by photon, km.
    #   light_time()        elapsed time from photon departure to arrival, sec.
    #   resolution()        resolution based on surface distance, as a quantity
    #                       defined perpendicular to the lines of sight, in
    #                       km/pixel.
    ############################################################################

    def distance(self, event_key, direction='dep'):
        """Distance in km between a photon's departure and its arrival.

        Input:
            event_key       key defining the surface event.
            direction       'arr' for distance traveled by the arriving photon;
                            'dep' for distance traveled by the departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert direction in ('dep', 'arr')

        key = ('distance', event_key, direction)
        if key not in self.backplanes:
            lt = self.light_time(event_key, direction)
            self.register_backplane(key, lt * constants.C)

        return self.backplanes[key]

    def light_time(self, event_key, direction='dep'):
        """Time in seconds between a photon's departure and its arrival.

        Input:
            event_key       key defining the surface event.
            direction       'arr' for the travel time of the arriving photon;
                            'dep' for the travel time of the departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert direction in ('dep', 'arr')

        key = ('light_time', event_key, direction)
        if key not in self.backplanes:
            if direction == 'arr':
                event = self.get_surface_event_with_arr(event_key)
            else:
                event = self.get_surface_event(event_key)

            lt = event.subfields[direction + '_lt']
            self.register_backplane(key, abs(lt))

        return self.backplanes[key]

    def event_time(self, event_key):
        """Absolute time in seconds TDB when the photon intercepted the surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)

        key = ('event_time', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event(event_key)
            self.register_backplane(key, event.time)

        return self.backplanes[key]

    def resolution(self, event_key, axis='u'):
        """Projected resolution in km/pixel at the surface intercept.

        Defined perpendicular to the line of sight.

        Input:
            event_key       key defining the surface event.
            axis            'u' for resolution along the horizontal axis of the
                                observation;
                            'v' for resolution along the vertical axis of the
                                observation.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert axis in ('u','v')

        key = ('resolution', event_key, axis)
        if key not in self.backplanes:
            distance = self.distance(event_key)

            (dlos_du, dlos_dv) = self.meshgrid.dlos_duv.as_columns()
            u_resolution = distance * dlos_du.as_vector3().norm()
            v_resolution = distance * dlos_dv.as_vector3().norm()

            self.register_backplane(key[:-1] + ('u',), u_resolution)
            self.register_backplane(key[:-1] + ('v',), v_resolution)

        return self.backplanes[key]

    ############################################################################
    # Basic body, path intercept versions
    #   center_distance()   distance traveled by photon, km.
    #   center_light_time() elapsed time from photon departure to arrival, sec.
    #   center_resolution() resolution based on body center distance, km/pixel.
    ############################################################################

    def center_distance(self, event_key, direction='obs'):
        """Distance traveled by a photon between paths.

        Input:
            event_key       key defining the event at the body's path.
            direction       'arr' or 'sun' to return the distance traveled by an
                                           arriving photon;
                            'dep' or 'obs' to return the distance traveled by a
                                           departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('center_distance', event_key, direction)
        if key not in self.backplanes:
            lt = self.center_light_time(event_key, direction)
            self.register_gridless_backplane(key, lt * constants.C)

        return self.backplanes[key]

    def center_light_time(self, event_key, direction='dep'):
        """Light travel time in seconds from a path.

        Input:
            event_key       key defining the event at the body's path.
            direction       'arr' or 'sun' to return the travel time for an
                                           arriving photon;
                            'dep' or 'obs' to return the travel time for a
                                           departing photon.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert direction in ('obs', 'sun', 'dep', 'arr')

        key = ('center_light_time', event_key, direction)
        if key not in self.backplanes:
            if direction in ('sun', 'arr'):
                event = self.get_gridless_event_with_arr(event_key)
                lt = event.arr_lt
            else:
                event = self.get_gridless_event(event_key)
                lt = event.dep_lt

            self.register_gridless_backplane(key, abs(lt))

        return self.backplanes[key]

    def center_time(self, event_key):
        """The absolute time when the photon intercepted the path.

        Measured in seconds TDB.

        Input:
            event_key       key defining the event at the body's path.
        """

        event_key = Backplane.standardize_event_key(event_key)

        key = ('center_time', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)
            self.register_gridless_backplane(key, event.time)

        return self.backplanes[key]

    def center_resolution(self, event_key, axis='u'):
        """Directionless projected spatial resolution in km/pixel.

        Measured at the central path of a body, based on range alone.

        Input:
            event_key       key defining the event at the body's path.
            axis            'u' for resolution along the horizontal axis of the
                                observation;
                            'v' for resolution along the vertical axis of the
                                observation.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert axis in ('u','v')

        key = ('center_resolution', event_key, axis)
        if key not in self.backplanes:
            distance = self.center_distance(event_key)

            (dlos_du, dlos_dv) = self.obs.fov.center_dlos_duv.as_columns()
            u_resolution = distance * dlos_du.as_vector3().norm()
            v_resolution = distance * dlos_dv.as_vector3().norm()

            self.register_gridless_backplane(key[:-1] + ('u',), u_resolution)
            self.register_gridless_backplane(key[:-1] + ('v',), v_resolution)

        return self.backplanes[key]

    ############################################################################
    # Lighting geometry, surface intercept version
    #   incidence_angle()   incidence angle at surface, radians.
    #   emission_angle()    emission angle at surface, radians.
    #   lambert_law()       Lambert Law model for surface, cos(incidence).
    ############################################################################

    def incidence_angle(self, event_key):
        """Incidence angle of the arriving photons at the local surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('incidence_angle', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            incidence = event.incidence_angle()

            # Ring incidence angles are always 0-90 degrees
            if event.surface.COORDINATE_TYPE == 'polar':

                # flip is True wherever incidence angle has to be changed
                flip = (incidence > constants.HALFPI)
                self.register_backplane(('ring_flip', event_key), flip)

                # Now flip incidence angles where necessary
                incidence = constants.PI * flip + (1. - 2.*flip) * incidence

            self.register_backplane(key, incidence)

        return self.backplanes[key]

    def emission_angle(self, event_key):
        """Emission angle of the departing photons at the local surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('emission_angle', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event(event_key)
            emission = event.emission_angle()

            # Ring emission angles are always measured from the lit side normal
            if event.surface.COORDINATE_TYPE == 'polar':

                # Get the flip flag
                ignore = self.incidence_angle(event_key)
                flip = self.backplanes[('ring_flip', event_key)]

                # Flip emission angles where necessary
                emission = constants.PI * flip + (1. - 2.*flip) * emission

            self.register_backplane(key, emission)

        return self.backplanes[key]

    def phase_angle(self, event_key):
        """Phase angle between the arriving and departing photons.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('phase_angle', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            self.register_backplane(key, event.phase_angle())

        return self.backplanes[key]

    def scattering_angle(self, event_key):
        """Scattering angle between the arriving and departing photons.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('scattering_angle', event_key)
        if key not in self.backplanes:
            self.register_backplane(key, constants.PI -
                                         self.phase_angle(event_key))

        return self.backplanes[key]

    def lambert_law(self, event_key):
        """Lambert law model cos(incidence_angle) for the surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('lambert_law', event_key)
        if key not in self.backplanes:
            incidence = self.incidence_angle(event_key)
            lambert_law = incidence.cos()
            lambert_law.mask |= (incidence.vals >= constants.HALFPI)
            lambert_law.vals[lambert_law.mask] = 0.
            self.register_backplane(key, lambert_law)

        return self.backplanes[key]

    ############################################################################
    # Lighting geometry, path intercept version
    #   center_incidence_angle()    incidence angle at path, radians. This uses
    #                               the z-axis of the target frame to define the
    #                               normal vector; it is used primarily for
    #                               rings.
    #   center_emission_angle()     emission angle at surface, radians. See
    #                               above.
    #   center_phase_angle()        phase angle at body's center, radians.
    #   center_scattering_angle()   scattering angle at body's center, radians.
    ############################################################################

    def center_incidence_angle(self, event_key):
        """Incidence angle of the arriving photons at the body's central path.

        This uses the z-axis of the body's frame to define the local normal.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('center_incidence_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)

            # Sign on event.arr is negative because photon is incoming
            latitude = (-event.arr.to_scalar(2) / event.arr.norm()).arcsin()
            incidence = constants.HALFPI - latitude

            # Ring incidence angles are always 0-90 degrees
            if event.surface.COORDINATE_TYPE == 'polar':

                # The flip is True wherever incidence angle has to be changed
                flip = (incidence > constants.HALFPI)
                self.register_gridless_backplane(('ring_center_flip',
                                                  event_key), flip)

                # Now flip incidence angles where necessary
                if flip.sum() > 0:
                    incidence = constants.PI * flip + (1. - 2.*flip) * incidence

            self.register_gridless_backplane(key, incidence)

        return self.backplanes[key]

    def center_emission_angle(self, event_key):
        """Emission angle of the departing photons at the body's central path.

        This uses the z-axis of the body's frame to define the local normal.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('center_emission_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)

            latitude = (event.dep.to_scalar(2) / event.dep.norm()).arcsin()
            emission = constants.HALFPI - latitude

            # Ring emission angles are always measured from the lit side normal
            if event.surface.COORDINATE_TYPE == 'polar':

                # Get the flip flag
                ignore = self.center_incidence_angle(event_key)
                flip = self.backplanes[('ring_center_flip', event_key)]

                # Flip emission angles where necessary
                if flip.sum() > 0:
                    emission = constants.PI * flip + (1. - 2.*flip) * emission

            self.register_gridless_backplane(key, emission)

        return self.backplanes[key]

    def center_phase_angle(self, event_key):
        """Phase angle as measured at the body's central path.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('center_phase_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)
            self.register_gridless_backplane(key, event.phase_angle())

        return self.backplanes[key]

    def center_scattering_angle(self, event_key):
        """Scattering angle as measured at the body's central path.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('center_scattering_angle', event_key)
        if key not in self.backplanes:
            angle = constants.PI - self.center_phase_angle(event_key)
            self.register_gridless_backplane(key, angle)

        return self.backplanes[key]

    ############################################################################
    # Body surface geometry, surface intercept versions
    #   longitude()
    #   latitude()
    ############################################################################

    def longitude(self, event_key, reference='iau', direction='west',
                                                    minimum=0):
        """Longitude at the surface intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
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
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in ('iau', 'sun', 'sha', 'obs', 'oha')
        assert direction in ('east', 'west')
        assert minimum in (0, -180)

        # Look up under the desired reference
        key0 = ('longitude', event_key)
        key = key0 + (reference, direction, minimum)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If it is not found with default keys, fill in those backplanes
        # Note that longitudes default to eastward for right-handed coordinates
        key_default = key0 + ('iau', 'east', 0)
        if not self.backplanes.has_key(key_default):
            self._fill_surface_intercepts(event_key)

        # Fill in the values for this key
        if reference == 'iau':
            ref_lon = 0.
        elif reference == 'sun':
            ref_lon = self.sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self.sub_solar_longitude(event_key) - constants.PI
        elif reference == 'obs':
            ref_lon = self.sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self.sub_observer_longitude(event_key) - constants.PI

        lon = self.backplanes[key_default] - ref_lon

        if direction == 'west': lon = -lon

        if minimum == 0:
            lon = lon % constants.TWOPI
        else:
            lon = (lon + constants.PI) % constants.TWOPI - constants.PI

        self.register_backplane(key, lon)
        return self.backplanes[key]

    def latitude(self, event_key, lat_type='centric'):
        """Latitude at the surface intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
            lat_type        defines the type of latitude measurement:
                            'centric'   for planetocentric;
                            'graphic'   for planetographic;
                            'squashed'  for an intermediate latitude type used
                                        internally.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert lat_type in ('centric', 'graphic', 'squashed')

        # Look up under the desired reference
        key0 = ('latitude', event_key)
        key = key0 + (lat_type,)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If it is not found with default keys, fill in those backplanes
        key_default = key0 + ('squashed',)
        if not self.backplanes.has_key(key_default):
            self._fill_surface_intercepts(event_key)

        # Fill in the values for this key
        lat = self.backplanes[key_default]
        if lat_type == 'squashed':
            return lat

        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE in ('spherical', 'limb')

        if lat_type == 'centric':
            lat = event.surface.lat_to_centric(lat)
        else:
            lat = event.surface.lat_to_graphic(lat)

        self.register_backplane(key, lat)
        return self.backplanes[key]

    def _fill_surface_intercepts(self, event_key):
        """Internal method to fill in the surface intercept geometry backplanes.
        """

        # Get the surface intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)

        # If this is actually a limb event, define the limb backplanes instead
        if event.surface.COORDINATE_TYPE == 'limb':
            self._fill_limb_intercepts(event_key)
            return

        assert event.surface.COORDINATE_TYPE == 'spherical'

        self.register_backplane(('longitude', event_key, 'iau', 'east', 0),
                                event.coord1)
        self.register_backplane(('latitude', event_key, 'squashed'),
                                event.coord2)

    ############################################################################
    # Surface geometry, path intercept versions
    #   sub_longitude()
    #   sub_solar_longitude()
    #   sub_observer_longitude()
    #   sub_latitude()
    #   sub_solar_latitude()
    #   sub_observer_latitude()
    ############################################################################

    def sub_longitude(self, event_key, reference='obs', direction='east'):
        """Sub-solar or sub-observer longitude at the body's path.

        Measured relative to the x-coordinate of the body's coordinate frame.
        This is the IAU meridian for planets and moons, and the J2000 ascending
        node for rings.

        Input:
            event_key       key defining the event on the body's path.
            reference       'obs' for the sub-observer longitude;
                            'sun' for the sub-solar longitude.
            direction       the direction of increasing longitude, either 'east'
                            or 'west'.
        """

        assert reference in ('obs', 'sun')

        if reference == 'obs':
            lon = self.sub_observer_longitude(event_key)
        else:
            lon = self.sub_solar_longitude(event_key)

        if direction == 'west':
            lon = constants.TWOPI - lon

        return lon

    def sub_solar_longitude(self, event_key):
        """Sub-solar longitude

        Primarily used internally. Longitudes are measured eastward from the
        x-axis of the target's coordinate frame.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('sub_solar_longitude', event_key)

        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)
            arr = -event.apparent_arr()
            lon = arr.to_scalar(1).arctan2(arr.to_scalar(0)) % constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

    def sub_observer_longitude(self, event_key):
        """Sub-observer longitude.

        Primarily used internally. Longitudes are measured eastward from the
        x-axis of the target's coordinate frame.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('sub_observer_longitude', event_key)

        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)
            dep = event.apparent_dep()
            lon = dep.to_scalar(1).arctan2(dep.to_scalar(0)) % constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

    def sub_latitude(self, event_key, reference='obs', lat_type='centric'):
        """Sub-solar or sub-observer latitude at the body's path.

        Input:
            event_key       key defining the event on the body's path.
            reference       'obs' for the sub-observer latitude;
                            'sun' for the sub-solar latitude.
            lat_type        'centric' for planetocentric latitude;
                            'graphic' for planetographic latitude.
        """

        assert reference in ('obs', 'sun')

        if reference == 'obs':
            return self.sub_observer_latitude(event_key, lat_type)
        else:
            return self.sub_solar_latitude(event_key, lat_type)

    def sub_solar_latitude(self, event_key, lat_type='centric'):
        """Sub-solar latitude, primarily used internally."""

        event_key = Backplane.standardize_event_key(event_key)
        assert lat_type in ('centric', 'graphic')

        key = ('sub_solar_latitude', event_key, lat_type)
        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)
            arr = -event.apparent_arr()

            if lat_type == 'graphic':
                arr = arr.element_mul(event.surface.unsquash_sq)

            lat = (arr.to_scalar(2) / arr.norm()).arcsin()
            self.register_gridless_backplane(key, lat)

        return self.backplanes[key]

    def sub_observer_latitude(self, event_key, lat_type='centric'):
        """Sub-observer latitude, primarily used internally."""

        event_key = Backplane.standardize_event_key(event_key)
        assert lat_type in ('centric', 'graphic')

        key = ('sub_observer_latitude', event_key, lat_type)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)
            dep = event.apparent_dep()

            if lat_type == 'graphic':
                dep = dep.element_mul(event.surface.unsquash_sq)

            lat = (dep.to_scalar(2) / dep.norm()).arcsin()
            self.register_gridless_backplane(key, lat)

        return self.backplanes[key]

    ############################################################################

    def pole_clock_angle(self, event_key):
        """Projected pole vector on the sky, measured from north through wes.

        In other words, measured clockwise on the sky."""

        event_key = Backplane.standardize_event_key(event_key)

        key = ('pole_clock_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)

            # Get the body frame's Z-axis in J2000 coordinates
            frame = Frame.J2000.wrt(event.frame)
            xform = frame.transform_at_time(event.time)
            pole_j2000 = xform.rotate(Vector3.ZAXIS)

            # Define the vector to the observer in the J2000 frame
            dep = event.aberrated_dep_ssb()

            # Construct a rotation matrix from J2000 to a frame in which the
            # Z-axis points along -dep and the J2000 pole is in the X-Z plane.
            # As it appears to the observer, the Z-axis points toward the body,
            # the X-axis points toward celestial north as projected on the sky,
            # and the Y-axis points toward celestial west (not east!).
            rotmat = Matrix3.twovec(-dep, 2, Vector3.ZAXIS, 0)

            # Rotate the body frame's Z-axis to this frame.
            pole = rotmat * pole_j2000

            # Convert the X and Y components of the rotated pole into an angle
            coords = pole.to_scalars()
            clock_angle = coords[1].arctan2(coords[0]) % constants.TWOPI

            self.register_gridless_backplane(key, clock_angle)

        return self.backplanes[key]

    def pole_position_angle(self, event_key):
        """Projected angle of a body's pole vector on the sky, measured from
        celestial north toward celestial east (i.e., counterclockwise on the
        sky)."""

        event_key = Backplane.standardize_event_key(event_key)

        key = ('pole_position_angle', event_key)
        if key not in self.backplanes:
            self.register_gridless_backplane(key,
                                constants.TWOPI - self.pole_clock_angle(event_key))

        return self.backplanes[key]

    ############################################################################
    # Surface resolution, surface intercepts only
    #   finest_resolution()
    #   coarsest_resolution()
    ############################################################################

    def finest_resolution(self, event_key):
        """Projected resolution in km/pixel for the optimal direction

        Determined a the intercept point on the surface.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('finest_resolution', event_key)
        if key not in self.backplanes:
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    def coarsest_resolution(self, event_key):
        """Projected spatial resolution in km/pixel in the worst direction at
        the intercept point.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('coarsest_resolution', event_key)
        if key not in self.backplanes:
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    def _fill_surface_resolution(self, event_key):
        """Internal method to fill in the surface resolution backplanes.
        """

        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event_w_derivs(event_key)

        dpos_duv = event.state.d_dlos.chain(self.meshgrid.dlos_duv)
        (minres, maxres) = Surface.resolution(dpos_duv)

        self.register_backplane(('finest_resolution', event_key), minres)
        self.register_backplane(('coarsest_resolution', event_key), maxres)

    ############################################################################
    # Limb geometry, surface intercepts only
    #   altitude()
    ############################################################################

#     def altitude(self, event_key):
#         """Elevation of a limb point above the body's surface.
# 
#         Input:
#             event_key       key defining the ring surface event.
#         """
# 
#         event_key = Backplane.standardize_event_key(event_key)
#         key = ('altitude', event_key)
#         if key not in self.backplanes:
#             self._fill_limb_intercepts(event_key)
# 
#         return self.backplanes[key]
# 
#     def _fill_limb_intercepts(self, event_key):
#         """Internal method to fill in the limb intercept geometry backplanes.
#         """
# 
#         # Get the limb intercept coordinates
#         event_key = Backplane.standardize_event_key(event_key)
#         event = self.get_surface_event(event_key)
#         assert event.surface.COORDINATE_TYPE == 'limb'
# 
#         self.register_backplane(('longitude', event_key, 'iau', 'east', 0),
#                                 event.coord1)
#         self.register_backplane(('latitude', event_key, 'squashed'),
#                                 event.coord2)
#         self.register_backplane(('altitude', event_key),
#                                 event.coord3)

    ############################################################################
    # Ring plane geometry, surface intercept version
    #   ring_radius()
    #   ring_longitude()
    #   ring_azimuth()
    #   ring_elevation()
    ############################################################################

    def ring_radius(self, event_key):
        """Radius of the ring intercept point in the observation.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ring_radius', event_key)
        if key not in self.backplanes:
            self._fill_ring_intercepts(event_key)

        return self.backplanes[key]

    def ring_longitude(self, event_key, reference='node'):
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
                            Put 'old-' in front of the last four for deprecated
                            definitions, which are slower but infinitesimally
                            more accurate.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha',
                             'old-obs', 'old-oha', 'old-sun', 'old-sha'}
                            # The last four are deprecated but not deleted

        # Look up under the desired reference
        key = ('ring_longitude', event_key, reference)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If it is not found with reference='node', fill in those backplanes
        key_node = key[:-1] + ('node',)
        if not self.backplanes.has_key(key_node):
            self._fill_ring_intercepts(event_key)

        # Now apply the reference longitude
        if reference == 'node':
            return self.backplanes[key]

        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self.sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self.sub_solar_longitude(event_key) - constants.PI
        elif reference == 'obs':
            ref_lon = self.sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self.sub_observer_longitude(event_key) - constants.PI

        # These four options are deprecated but not deleted. The above versions
        # are much simpler and faster and the difference is infinitesimal.
        elif reference == 'old-sun':
            ref_lon = self._sub_solar_ring_longitude(event_key)
        elif reference == 'old-sha':
            ref_lon = self._sub_solar_ring_longitude(event_key) - constants.PI
        elif reference == 'old-obs':
            ref_lon = self._sub_observer_ring_longitude(event_key)
        elif reference == 'old-oha':
            ref_lon = self._sub_observer_ring_longitude(event_key) -constants.PI

        lon = (self.backplanes[key_node] - ref_lon) % constants.TWOPI
        self.register_backplane(key, lon)

        return self.backplanes[key]

    def _aries_ring_longitude(self, event_key):
        """Longitude of First Point of Aries from the ring ascending node.

        Primarily used internally. Longitudes are measured in the eastward
        (prograde) direction.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('_aries_ring_longitude', event_key)

        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)
            frame = Frame.as_primary_frame(event.frame)
            lon = (-frame.node_at_time(event.time)) % constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

    def ring_azimuth(self, event_key, reference='obs'):
        """Angle from a reference direction to the local radial direction.

        The angle is measured in the prograde direction from a reference
        direction to the local radial, as measured at the ring intercept point
        and projected into the ring plane. This value is 90 at the left ansa and
        270 at the right ansa."

        The reference direction can be 'obs' for the apparent departing
        direction of the photon, or 'sun' for the (negative) apparent direction
        of the arriving photon.

        Input:
            event_key       key defining the ring surface event.
            reference       'obs' or 'sun'; see discussion above.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in ('obs', 'sun')

        # Look up under the desired reference
        key = ('ring_azimuth', event_key, reference)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If not found, fill in the ring events if necessary
        if not self.backplanes.has_key(('ring_radius', event_key)):
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
        az = (rad_angle - ref_angle) % constants.TWOPI
        self.register_backplane(key, az)

        return self.backplanes[key]

    def ring_elevation(self, event_key, reference='obs'):
        """Angle from the ring plane and the photon direction.

        Evaluated at the ring intercept point. The angle is positive on the side
        of the ring plane where rotation is prograde; negative on the opposite
        side.

        The reference direction can be 'obs' for the apparent departing
        direction of the photon, or 'sun' for the (negative) apparent direction
        of the arriving photon.

        Input:
            event_key       key defining the ring surface event.
            reference       'obs' or 'sun'; see discussion above.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in ('obs', 'sun')

        # Look up under the desired reference
        key = ('ring_elevation', event_key, reference)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If not found, fill in the ring events if necessary
        if not self.backplanes.has_key(('ring_radius', event_key)):
            self._fill_ring_intercepts(event_key)

        # reference = 'obs'
        if reference == 'obs':
            event = self.get_surface_event(event_key)
            dir = event.apparent_dep()

        # reference = 'sun'
        else:
            event = self.get_surface_event_with_arr(event_key)
            dir = -event.apparent_arr()

        el = constants.HALFPI - event.perp.sep(dir)
        self.register_backplane(key, el)

        return self.backplanes[key]

    def _fill_ring_intercepts(self, event_key):
        """Internal method to fill in the ring intercept geometry backplanes.
        """

        # Get the ring intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == 'polar'

        self.register_backplane(('ring_radius', event_key), event.coord1)
        self.register_backplane(('ring_longitude', event_key, 'node'),
                                event.coord2)

    # Deprecated...
    def _sub_observer_ring_longitude(self, event_key):
        """Sub-observer longitude, evaluated at the ring intercept time.

        Used only internally. DEPRECATED.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('_sub_observer_ring_longitude', event_key)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (Vector3.ZERO, Vector3.ZERO),
                                         event.origin, event.frame)
        center_event = self.obs_event.origin.photon_from_event(center_event)[1]

        surface = Backplane.get_surface(event_key[0])
        assert surface.COORDINATE_TYPE == 'polar'
        (r,lon) = surface.coords_from_vector3(center_event.apparent_dep(),
                                              axes=2)

        self.register_backplane(key, lon)
        return self.backplanes[key]

    # Deprecated...
    def _sub_solar_ring_longitude(self, event_key):
        """Sub-solar longitude at the ring center, but evaluated at the ring
        intercept time. Used only internally. DEPRECATED.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('_sub_solar_ring_longitude', event_key)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (Vector3.ZERO, Vector3.ZERO),
                                         event.origin, event.frame)
        center_event = AliasPath("SUN").photon_to_event(center_event)[1]
        surface = Backplane.get_surface(event_key[0])
        assert event.surface.COORDINATE_TYPE == 'polar'
        (r,lon) = surface.coords_from_vector3(-center_event.apparent_arr(),
                                              axes=2)

        self.register_backplane(key, lon)
        return self.backplanes[key]

    ############################################################################
    # Ring plane lighting geometry, surface intercept version
    #   ring_incidence_angle()
    #   ring_emission_angle()
    ############################################################################

    def ring_incidence_angle(self, event_key, pole='sunward'):
        """Incidence angle of the arriving photons at the local ring surface.

        By default, angles are measured from the sunward pole and should always
        be <= pi/2. However, calculations for values relative to the IAU-defined
        north pole and relative to the prograde pole are also supported.

        Input:
            event_key       key defining the ring surface event.
            pole            'sunward' for the ring pole on the illuminated face;
                            'north' for the pole on the IAU-defined north face;
                            'prograde' for the pole defined by the direction of
                                positive angular momentum.
        """

        assert pole in {'sunward', 'north', 'prograde'}

        # The sunward pole uses the standard definition of incidence angle
        if pole == 'sunward':
            return self.incidence_angle(event_key)

        event_key = Backplane.standardize_event_key(event_key)

        # Return the cached copy if it exists
        key = ('ring_incidence_angle', event_key, pole)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # Derive the prograde incidence angle if necessary
        key_prograde = key[:-1] + ('prograde',)
        if not self.backplanes.has_key(key_prograde):
            event = self.get_surface_event_with_arr(event_key)
            incidence = event.incidence_angle()
            self.register_backplane(key_prograde, incidence)

        if pole == 'prograde':
            return self.backplanes[key_prograde]

        # If the ring is prograde, 'north' and 'prograde' are the same
        body = Body.lookup(event_key[0])
        if not body.ring_is_retrograde:
            return self.backplanes[prograde_key]

        # Otherwise, flip the incidence angles and return a new backplane
        incidence = constants.PI - self.backplanes[key_prograde]
        self.register_backplane(key, incidence)
        return self.backplanes[key]

    def ring_emission_angle(self, event_key, pole='sunward'):
        """Emission angle of the departing photons at the local ring surface.

        By default, angles are measured from the sunward pole, so the emission
        angle should be < pi/2 on the sunlit side and > pi/2 on the dark side
        of the rings. However, calculations for values relative to the
        IAU-defined north pole and relative to the prograde pole are also
        supported.

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

        event_key = Backplane.standardize_event_key(event_key)

        # Return the cached copy if it exists
        key = ('ring_emission_angle', event_key, pole)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # Derive the prograde emission angle if necessary
        key_prograde = key[:-1] + ('prograde',)
        if not self.backplanes.has_key(key_prograde):
            event = self.get_surface_event(event_key)
            emission = event.emission_angle()
            self.register_backplane(key_prograde, emission)

        if pole == 'prograde' :
            return self.backplanes[key_prograde]

        # If the ring is prograde, 'north' and 'prograde' are the same
        body = Body.lookup(event_key[0])
        if not body.ring_is_retrograde:
            return self.backplanes[key_prograde]

        # Otherwise, flip the emission angles and return a new backplane
        emission = constants.PI - self.backplanes[key_prograde]
        self.register_backplane(key, emission)
        return self.backplanes[key]

    ############################################################################
    # Ring plane geometry, path intercept versions
    #   sub_ring_longitude()
    #   ring_center_incidence_angle()
    #   ring_center_emission_angle()
    ############################################################################

    def sub_ring_longitude(self, event_key, reference='obs', origin='node'):
        """Sub-solar or sub-observer longitude in the ring plane.

        It can be defined relative to the ring plane's J2000 ascending node or
        the First Point of Aries.

        Input:
            event_key       key defining the event on the center of the ring's
                            path.
            reference       'obs' for the sub-observer longitude;
                            'sun' for the sub-solar longitude.
            origin          'node' for the longitude relative to the J2000
                            ascending node of the ring plane; 'aries' for the
                            longitude relative to the First Point of Aries.
        """

        assert reference in ('obs', 'sun')
        assert origin in ('node', 'aries')

        # Longitudes relative to the node use the standard definition
        if reference == 'obs':
            lon = self.sub_observer_longitude(event_key)
        else:
            lon = self.sub_solar_longitude(event_key)

        event_key = Backplane.standardize_event_key(event_key)

        # Return a cached backplane if it exists
        key = ('sub_ring_longitude', event_key, reference, origin)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # Otherwise, create the backplane
        if origin == 'aries':
            lon = (lon - self._aries_ring_longitude(event_key)) % constants.TWOPI
            self.register_gridless_backplane(key, lon)

        return lon

    def ring_center_incidence_angle(self, event_key, pole='sunward'):
        """Incidence angle of the arriving photons at the ring system center.

        By default, angles are measured from the sunward pole and should always
        be <= pi/2. However, calculations for values relative to the IAU-defined
        north pole and relative to the prograde pole are also supported.

        Input:
            event_key       key defining the ring surface event.
            pole            'sunward' for the ring pole on the illuminated face;
                            'north' for the pole on the IAU-defined north face;
                            'prograde' for the pole defined by the direction of
                                positive angular momentum.
        """

        assert pole in {'sunward', 'north', 'prograde'}

        # The sunward pole uses the standard definition of incidence angle
        if pole == 'sunward':
            return self.center_incidence_angle(event_key)

        event_key = Backplane.standardize_event_key(event_key)

        # Return the cached copy if it exists
        key = ('ring_center_incidence_angle', event_key, pole)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # Derive the prograde incidence angle if necessary
        key_prograde = key[:-1] + ('prograde',)
        if not self.backplanes.has_key(key_prograde):
            event = self.get_gridless_event_with_arr(event_key)

            # Sign on event.arr is negative because photon is incoming
            latitude = (-event.arr.to_scalar(2) / event.arr.norm()).arcsin()
            incidence = constants.HALFPI - latitude

            self.register_gridless_backplane(key, incidence)

        if pole == 'prograde':
            return self.backplanes[key_prograde]

        # If the ring is prograde, 'north' and 'prograde' are the same
        body = Body.lookup(event_key[0])
        if not body.ring_is_retrograde:
            return self.backplanes[prograde_key]

        # Otherwise, flip the incidence angle and return a new backplane
        incidence = constants.PI - self.backplanes[key_prograde]
        self.register_gridless_backplane(key, incidence)

        return self.backplanes[key]

    def ring_center_emission_angle(self, event_key, pole='sunward'):
        """Emission angle of departing photons at the center of the ring system.

        By default, angles are measured from the sunward pole, so the emission
        angle should be < pi/2 on the sunlit side and > pi/2 on the dark side
        of the rings. However, calculations for values relative to the
        IAU-defined north pole and relative to the prograde pole are also
        supported.

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

        event_key = Backplane.standardize_event_key(event_key)

        # Return the cached copy if it exists
        key = ('ring_center_emission_angle', event_key, pole)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # Derive the prograde emission angle if necessary
        key_prograde = key[:-1] + ('prograde',)
        if not self.backplanes.has_key(key_prograde):
            event = self.get_gridless_event(event_key)

            latitude = (event.dep.to_scalar(2) / event.dep.norm()).arcsin()
            emission = constants.HALFPI - latitude

            self.register_gridless_backplane(key, emission)

        if pole == 'prograde':
            return self.backplanes[key_prograde]

        # If the ring is prograde, 'north' and 'prograde' are the same
        body = Body.lookup(event_key[0])
        if not body.ring_is_retrograde:
            return self.backplanes[prograde_key]

        # Otherwise, flip the emission angle and return a new backplane
        emission = constants.PI - self.backplanes[key_prograde]
        self.register_backplane(key, emission)

        return self.backplanes[key]

    ############################################################################
    # Ring plane geometry, surface intercept only
    #   ring_radial_resolution()
    #   ring_angular_resolution()
    ############################################################################

    def ring_radial_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring intercept point.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ring_radial_resolution', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == 'polar'

        rad = event.coord1
        drad_duv = rad.d_dlos.chain(self.meshgrid.dlos_duv)
        res = drad_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    def ring_angular_resolution(self, event_key):
        """Projected angular resolution in radians/pixel at the ring intercept.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ring_angular_resolution', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == 'polar'

        lon = event.coord2
        dlon_duv = lon.d_dlos.chain(self.meshgrid.dlos_duv)
        res = dlon_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    ############################################################################
    # Ring ansa geometry, surface intercept only
    #   ansa_radius()
    #   ansa_altitude()
    #   ansa_longitude()
    ############################################################################

    def ansa_radius(self, event_key, radius_type='right'):
        """Radius of the ring ansa intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
        """

        # Look up under the desired radius type
        event_key = Backplane.standardize_event_key(event_key)
        key0 = ('ansa_radius', event_key)
        key = key0 + (radius_type,)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If not found, look up the default 'right'
        assert radius_type in ('right', 'left', 'positive')

        key_default = key0 + ('right',)
        if not self.backplanes.has_key(key_default):
            self._fill_ansa_intercepts(event_key)

            backplane = self.backplanes[key_default]
            if radius_type == 'left':
                backplane = -backplane
            else:
                backplane = abs(backplane)

            self.register_backplane(key, backplane)

        return self.backplanes[key]

    def ansa_altitude(self, event_key):
        """Elevation of the ring ansa intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ansa_altitude', event_key)
        if key not in self.backplanes:
            self._fill_ansa_intercepts(event_key)

        return self.backplanes[key]

    def ansa_longitude(self, event_key, reference='node'):
        """Longitude of the ansa intercept point in the image.

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
                            Put 'old-' in front of the last four for deprecated
                            definitions, which are slower but infinitesimally
                            more accurate.
        """

        event_key = Backplane.standardize_event_key(event_key)
        assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha',
                             'old-obs', 'old-oha', 'old-sun', 'old-sha'}
                            # The last four are deprecated but not deleted

        # Look up under the desired reference
        key0 = ('ansa_longitude', event_key)
        key = key0 + (reference,)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # If it is not found with reference J2000, fill in those backplanes
        key_node = key0 + ('node',)
        if not self.backplanes.has_key(key_node):
            self._fill_ansa_longitudes(event_key)

        # Now apply the reference longitude
        if reference == 'node':
            return self.backplanes[key]

        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self.sub_solar_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self.sub_solar_longitude(event_key) - constants.PI
        elif reference == 'obs':
            ref_lon = self.sub_observer_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self.sub_observer_longitude(event_key) - constants.PI

        # These four options are deprecated but not deleted. The above versions
        # are much simpler and faster and the difference is infinitesimal.
        elif reference == 'old-sun':
            ref_lon = self._sub_solar_ansa_longitude(event_key)
        elif reference == 'old-sha':
            ref_lon = self._sub_solar_ansa_longitude(event_key) - constants.PI
        elif reference == 'old-obs':
            ref_lon = self._sub_observer_ansa_longitude(event_key)
        elif reference == 'old-oha':
            ref_lon = self._sub_observer_ansa_longitude(event_key) -constants.PI

        lon = (self.backplanes[key_node] - ref_lon) % constants.TWOPI
        self.register_backplane(key, lon)

        return self.backplanes[key]

    def _fill_ansa_intercepts(self, event_key):
        """Internal method to fill in the ansa intercept geometry backplanes.
        """

        # Get the ansa intercept coordinates
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        r = event.coord1
        z = event.coord2

        self.register_backplane(('ansa_radius', event_key, 'right'), r)
        self.register_backplane(('ansa_altitude', event_key), z)

    def _fill_ansa_longitudes(self, event_key):
        """Internal method to fill in the ansa intercept longitude backplane.
        """

        # Get the ansa intercept event
        event_key = Backplane.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        # Get the longitude in the associated ring plane
        lon = event.surface.ringplane.coords_from_vector3(event.pos, axes=2)[1]
        self.register_backplane(('ansa_longitude', event_key, 'node'), lon)

    # Deprecated
    def _sub_observer_ansa_longitude(self, event_key):
        """Sub-observer longitude evaluated at the ansa intercept time.

        Used only internally. DEPRECATED.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('_sub_observer_ansa_longitude', event_key)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (Vector3.ZERO, Vector3.ZERO),
                                         event.origin, event.frame)
        center_event = self.obs_event.origin.photon_from_event(center_event)[1]

        surface = Backplane.get_surface(event_key[0]).ringplane
        (r,lon) = surface.coords_from_vector3(center_event.apparent_dep(),
                                              axes=2)

        self.register_backplane(key, lon)
        return self.backplanes[key]

    # Deprecated
    def _sub_solar_ansa_longitude(self, event_key):
        """Sub-solar longitude evaluated at the ansa intercept time.

        Used only internally. DEPRECATED.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('_sub_solar_ansa_longitude', event_key)
        if self.backplanes.has_key(key):
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, (Vector3.ZERO, Vector3.ZERO),
                                         event.origin, event.frame)
        center_event = AliasPath('SUN').photon_to_event(center_event)[1]

        surface = Backplane.get_surface(event_key[0]).ringplane
        (r,lon) = surface.coords_from_vector3(-center_event.apparent_arr(),
                                              axes=2)

        self.register_backplane(key, lon)
        return self.backplanes[key]

    ############################################################################
    # Ansa resolution, surface intercepts only
    #
    # Radius keys: ('ansa_radial_resolution', event_key)
    # Elevation keys: ('ansa_vertical_resolution', event_key)
    ############################################################################

    def ansa_radial_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring ansa intercept.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ansa_radial_resolution', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        r = event.coord1
        dr_duv = r.d_dlos.chain(self.meshgrid.dlos_duv)
        res = dr_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    def ansa_vertical_resolution(self, event_key):
        """Projected radial resolution in km/pixel at the ring ansa intercept.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = Backplane.standardize_event_key(event_key)
        key = ('ansa_vertical_resolution', event_key)
        try:
            return self.backplanes[key]
        except KeyError: pass

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        z = event.coord2
        dz_duv = z.d_dlos.chain(self.meshgrid.dlos_duv)
        res = dz_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    ############################################################################
    # Masks
    ############################################################################

    def where_intercepted(self, event_key):
        """A mask where the surface was intercepted.
        """

        event_key  = Backplane.standardize_event_key(event_key)
        key = ('where_intercepted', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event(event_key)
            mask = self.mask_as_boolean(np.logical_not(event.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_inside_shadow(self, event_key, shadow_body):
        """A mask where the surface is in the shadow of a second body."""

        event_key = Backplane.standardize_event_key(event_key)
        shadow_body = Backplane.standardize_event_key(shadow_body)
        key = ('where_inside_shadow', event_key, shadow_body[0])
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            mask = self.mask_as_boolean(np.logical_not(event.mask) &
                                       np.logical_not(shadow_event.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_outside_shadow(self, event_key, shadow_body):
        """A mask where the surface is outside the shadow of a second body."""

        event_key  = Backplane.standardize_event_key(event_key)
        shadow_body = Backplane.standardize_event_key(shadow_body)
        key = ('where_outside_shadow', event_key, shadow_body[0])
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            shadow_event = self.get_surface_event(shadow_body + event_key)
            mask = self.mask_as_boolean(np.logical_not(event.mask) &
                                       shadow_event.mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_in_front(self, event_key, back_body):
        """A mask where the first surface is in front of the second surface.

        This is also True where the back_body is not behind the front body at 
        all."""

        event_key = Backplane.standardize_event_key(event_key)
        back_body  = Backplane.standardize_event_key(back_body)
        key = ('where_in_front', event_key, back_body[0])
        if key not in self.backplanes:

            # A surface is in front if it is unmasked and the second surface is
            # either masked or further away.
            front_unmasked = np.logical_not(
                                    self.get_surface_event(event_key).mask)
            back_masked = self.get_surface_event(back_body).mask
            mask = self.mask_as_boolean(front_unmasked & (back_masked |
                                            (self.distance(event_key).vals <
                                             self.distance(back_body).vals)))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_in_back(self, event_key, front_body):
        """A mask where the first surface is behind the second surface."""

        event_key = Backplane.standardize_event_key(event_key)
        front_body = Backplane.standardize_event_key(front_body)

        key = ('where_in_back', event_key, front_body[0])
        if key not in self.backplanes:

            # A surface is in back if it is unmasked and the second surface is
            # both unmasked and closer.
            back_unmasked  = np.logical_not(
                                    self.get_surface_event(event_key).mask)
            front_unmasked = np.logical_not(
                                    self.get_surface_event(front_body).mask)
            mask = self.mask_as_boolean(back_unmasked & front_unmasked &
                                    (self.distance(event_key).vals >
                                     self.distance(front_body).vals))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_sunward(self, event_key):
        """A mask where the surface of a body is facing toward the Sun."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ('where_sunward',) + event_key
        if key not in self.backplanes:
            incidence = self.incidence_angle(event_key)
            mask = self.mask_as_boolean((incidence.vals <= constants.HALFPI) &
                                       np.logical_not(incidence.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_antisunward(self, event_key):
        """A mask where the surface of a body is facing away fron the Sun."""

        event_key = Backplane.standardize_event_key(event_key)
        key = ('where_antisunward',) + event_key
        if key not in self.backplanes:
            incidence = self.incidence_angle(event_key)
            mask = self.mask_as_boolean((incidence.vals > constants.HALFPI) &
                                       np.logical_not(incidence.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    ############################################################################
    # Masks derived from backplanes
    ############################################################################

    def where_below(self, backplane_key, value):
        """A mask where the backplane is <= the specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_below', backplane_key, value)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            mask = self.mask_as_boolean((backplane.vals <= value) &
                                        np.logical_not(backplane.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_above(self, backplane_key, value):
        """A mask where the backplane is >= the specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_above', backplane_key, value)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            mask = self.mask_as_boolean((backplane.vals >= value) &
                                        np.logical_not(backplane.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_between(self, backplane_key, low, high):
        """A mask where the backplane is between the given values, inclusive."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_between', backplane_key, low, high)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            mask = self.mask_as_boolean((backplane.vals >= low) &
                                        (backplane.vals <= high) &
                                        np.logical_not(backplane.mask))
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_not(self, backplane_key):
        """A mask where the value of the given backplane is False."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_not', backplane_key)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            self.register_backplane(key, ~backplane)

        return self.backplanes[key]

    def where_any(self, *backplane_keys):
        """A mask where any of the given backplanes is True."""

        key = ('where_any',) + backplane_keys
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_keys[0]).copy()
            for next_mask in backplane_keys[1:]:
                backplane |= self.evaluate(next_mask)

            self.register_backplane(key, backplane)

        return self.backplanes[key]

    def where_all(self, *backplane_keys):
        """A mask where all of the given backplanes are True."""

        key = ('where_all',) + backplane_keys
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_keys[0]).copy()
            for next_mask in backplane_keys[1:]:
                backplane &= self.evaluate(next_mask)

            self.register_backplane(key, backplane)

        return self.backplanes[key]

    ############################################################################
    # Borders
    ############################################################################

    def _border_above_or_below(self, sign, backplane_key, value):
        """The locus of points <= or >= a specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)

        if sign > 0:
            key = ('border_above', backplane_key, value)
        else:
            key = ('border_below', backplane_key, value)

        if key not in self.backplanes:
            backplane = sign * (self.evaluate(backplane_key) - value)
            border = np.zeros(self.meshgrid.shape, dtype='bool')

            axes = len(backplane.shape)
            for axis in range(axes):
                xbackplane = backplane.vals.swapaxes(0, axis)
                xborder = border.swapaxes(0, axis)

                xborder[:-1] |= ((xbackplane[:-1] >= 0) &
                                 (xbackplane[1:]  < 0))
                xborder[1:]  |= ((xbackplane[1:]  >= 0) &
                                 (xbackplane[:-1] < 0))

            self.register_backplane(key, Boolean(border &
                                                np.logical_not(backplane.mask)))

        return self.backplanes[key]

    def border_above(self, backplane_key, value):
        """The locus of points surrounding the region >= a specified value."""

        return self._border_above_or_below(+1, backplane_key, value)

    def border_below(self, backplane_key, value):
        """The locus of points surrounding the region <= a specified value."""

        return self._border_above_or_below(-1, backplane_key, value)

    def border_atop(self, backplane_key, value):
        """The locus of points straddling the points closest to a border.

        This backplane is True for the pixels that fall closest to the
        transition from below to above.
        """

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('border_atop', backplane_key, value)
        if key not in self.backplanes:
            absval = self.evaluate(backplane_key) - value
            sign = absval.sign()
            absval = absval * sign

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
        """The locus of points that fall on the outer edge of a mask.

        "Outside" (value = False) identifies the first False pixels outside each
        area of True pixels; "Inside" (value = True) identifies the last True
        pixels adjacent to an area of False pixels."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)

        if value:
            key = ('border_inside', backplane_key)
        else:
            key = ('border_outside', backplane_key)

        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)

            # Reverse the backplane if value is False
            if not value:
                backplane = ~backplane

            border = np.zeros(backplane.shape, dtype='bool')

            axes = len(backplane.shape)
            for axis in range(axes):
                xbackplane = backplane.vals.swapaxes(0, axis)
                xborder = border.swapaxes(0, axis)

                xborder[:-1] |= ((xbackplane[:-1] ^ xbackplane[1:]) &
                                  xbackplane[:-1])
                xborder[1:]  |= ((xbackplane[1:] ^ xbackplane[:-1]) &
                                  xbackplane[1:])

            self.register_backplane(key, Boolean(border))

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
        'right_ascension', 'declination',
        'celestial_north_angle', 'celestial_east_angle',
        'center_right_ascension', 'center_declination',

        'distance', 'light_time', 'event_time', 'resolution',
        'center_distance', 'center_light_time', 'center_time',
            'center_resolution',

        'incidence_angle', 'emission_angle', 'phase_angle',
            'scattering_angle', 'lambert_law',
        'center_incidence_angle', 'center_emission_angle', 'center_phase_angle',
            'center_scattering_angle',

        'longitude', 'latitude',
        'sub_longitude', 'sub_latitude',
        'sub_solar_longitude', 'sub_observer_longitude',
        'sub_solar_latitude', 'sub_observer_latitude',
        'pole_clock_angle', 'pole_position_angle',
        'finest_resolution', 'coarsest_resolution',
        'altitude',

        'ring_radius', 'ring_longitude', 'ring_azimuth', 'ring_elevation',
        'ring_radial_resolution', 'ring_angular_resolution',
        'ring_incidence_angle', 'ring_emission_angle',
        'sub_ring_longitude',
        'ring_center_incidence_angle', 'ring_center_emission_angle',

        'ansa_radius', 'ansa_altitude', 'ansa_longitude',
        'ansa_radial_resolution', 'ansa_vertical_resolution',

        'where_intercepted',
        'where_inside_shadow', 'where_outside_shadow',
        'where_in_front', 'where_in_back',
        'where_sunward', 'where_antisunward',
        'where_below', 'where_above', 'where_between',
        'where_not', 'where_any', 'where_all',

        'border_below', 'border_above', 'border_atop',
        'border_inside', 'border_outside'}

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
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

UNITTEST_PRINT = True
UNITTEST_LOGGING = True
UNITTEST_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY, "cassini/ISS/W1573721822_1.IMG")
UNITTEST_UNDERSAMPLE = 16

def show_info(title, array):
    """Internal method to print summary information and display images as
    desired."""

    global UNITTEST_PRINT
    if not UNITTEST_PRINT: return

    print title

    # Mask summary
    if array.vals.dtype == np.dtype('bool'):
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

        Path.reset_registry()
        Frame.reset_registry()

        import oops.inst.cassini.iss as iss

        if UNITTEST_LOGGING: config.LOGGING.on("        ")

        snap = iss.from_file(UNITTEST_FILESPEC)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)

        bp = Backplane(snap, meshgrid)

        test = bp.right_ascension(apparent=False)
        show_info("Right ascension (deg)", test * constants.DPR)

        test = bp.right_ascension(apparent=True)
        show_info("Right ascension (deg, apparent)", test * constants.DPR)

        test = bp.declination(apparent=False)
        show_info("Declination (deg)", test * constants.DPR)

        test = bp.declination(apparent=True)
        show_info("Declination (deg, apparent)", test * constants.DPR)

        test = bp.distance('saturn')
        show_info("Range to Saturn (km)", test)

        test = bp.distance(('sun','saturn'))
        show_info("Saturn distance to Sun (km)", test)

        test = bp.light_time('saturn')
        show_info("Light-time from Saturn (sec)", test)

        test = bp.incidence_angle('saturn')
        show_info("Saturn incidence angle (deg)", test * constants.DPR)

        test = bp.emission_angle('saturn')
        show_info("Saturn emission angle (deg)", test * constants.DPR)

        test = bp.phase_angle('saturn')
        show_info("Saturn phase angle (deg)", test * constants.DPR)

        test = bp.scattering_angle('saturn')
        show_info("Saturn scattering angle (deg)", test * constants.DPR)

        test = bp.longitude('saturn')
        show_info("Saturn longitude (deg)", test * constants.DPR)

        test = bp.longitude('saturn', direction='west')
        show_info("Saturn longitude westward (deg)", test * constants.DPR)

        test = bp.longitude('saturn', minimum=-180)
        show_info("Saturn longitude with -180 minimum (deg)",
                                                    test * constants.DPR)

        test = bp.longitude('saturn', reference='iau')
        show_info("Saturn longitude wrt IAU frame (deg)",
                                                    test * constants.DPR)

        test = bp.longitude('saturn', reference='sun')
        show_info("Saturn longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.longitude('saturn', reference='sha')
        show_info("Saturn longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.longitude('saturn', reference='obs')
        show_info("Saturn longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.longitude('saturn', reference='oha')
        show_info("Saturn longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.latitude('saturn', lat_type='centric')
        show_info("Saturn geocentric latitude (deg)", test * constants.DPR)

        test = bp.latitude('saturn', lat_type='graphic')
        show_info("Saturn geographic latitude (deg)", test * constants.DPR)

        test = bp.finest_resolution('saturn')
        show_info("Saturn finest surface resolution (km)", test)

        test = bp.coarsest_resolution('saturn')
        show_info("Saturn coarsest surface resolution (km)", test)

        test = bp.ring_radius('saturn_main_rings')
        show_info("Ring radius (km)", test)

        test = bp.ring_radius('saturn_main_rings').without_mask()
        show_info("Ring radius unmasked (km)", test)

        test = bp.ring_longitude('saturn_main_rings',reference='node')
        show_info("Ring longitude wrt node (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='sun')
        show_info("Ring longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='sha')
        show_info("Ring longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='obs')
        show_info("Ring longitude wrt observer (deg)",
                                                    test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='oha')
        show_info("Ring longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.distance('saturn_main_rings')
        show_info("Range to rings (km)", test)

        test = bp.light_time('saturn_main_rings')
        show_info("Light time from rings (sec)", test)

        test = bp.distance(('sun', 'saturn_main_rings'))
        show_info("Ring distance to Sun (km)", test)

        test = bp.incidence_angle('saturn_main_rings')
        show_info("Ring incidence angle (deg)", test * constants.DPR)

        test = bp.emission_angle('saturn_main_rings')
        show_info("Ring emission angle (deg)", test * constants.DPR)

        test = bp.phase_angle('saturn_main_rings')
        show_info("Ring phase angle (deg)", test * constants.DPR)

        test = bp.ring_radial_resolution('saturn_main_rings')
        show_info("Ring radial resolution (km/pixel)", test)

        test = bp.ring_angular_resolution('saturn_main_rings')
        show_info("Ring angular resolution (deg/pixel)", test * constants.DPR)

        test = bp.where_intercepted('saturn')
        show_info("Saturn intercepted mask", test)

        test = bp.where_in_front('saturn', 'saturn_main_rings')
        show_info("Saturn in front of rings", test)

        test = bp.where_in_back('saturn', 'saturn_main_rings')
        show_info("Saturn in front of rings", test)

        test = bp.where_inside_shadow('saturn', 'saturn_main_rings')
        show_info("Saturn in front of rings", test)

        test = bp.where_outside_shadow('saturn', 'saturn_main_rings')
        show_info("Saturn in front of rings", test)

        test = bp.where_sunward('saturn')
        show_info("Saturn sunward", test)

        test = bp.where_antisunward('saturn')
        show_info("Saturn anti-sunward", test)

        test = bp.where_intercepted('saturn_main_rings')
        show_info("Rings intercepted mask", test)

        test = bp.where_in_front('saturn_main_rings', 'saturn')
        show_info("Rings in front of Saturn", test)

        test = bp.where_in_back('saturn_main_rings', 'saturn')
        show_info("Rings in front of Saturn", test)

        test = bp.where_inside_shadow('saturn_main_rings', 'saturn')
        show_info("Rings in front of Saturn", test)

        test = bp.where_outside_shadow('saturn_main_rings', 'saturn')
        show_info("Rings in front of Saturn", test)

        test = bp.where_sunward('saturn_main_rings')
        show_info("Rings sunward", test)

        test = bp.where_antisunward('saturn_main_rings')
        show_info("Rings anti-sunward", test)

        test = bp.right_ascension()
        show_info("Right ascension the old way (deg)", test * constants.DPR)

        test = bp.evaluate('right_ascension')
        show_info("Right ascension via evaluate() (deg)", test * constants.DPR)

        test = bp.where_intercepted('saturn')
        show_info("Saturn intercepted the old way", test)

        test = bp.evaluate(('where_intercepted', 'saturn'))
        show_info("Saturn where intercepted via evaluate()", test)

        test = bp.where_sunward('saturn')
        show_info("Saturn sunward the old way", test)

        test = bp.evaluate(('where_sunward', 'saturn'))
        show_info("Saturn sunward via evaluate()", test)

        test = bp.where_below(('incidence_angle', 'saturn'), constants.HALFPI)
        show_info("Saturn sunward via where_below()", test)

        test = bp.evaluate(('where_antisunward', 'saturn'))
        show_info("Saturn antisunward via evaluate()", test)

        test = bp.where_above(('incidence_angle', 'saturn'), constants.HALFPI)
        show_info("Saturn antisunward via where_above()", test)

        test = bp.where_between(('incidence_angle', 'saturn'), constants.HALFPI, 3.2)
        show_info("Saturn antisunward via where_between()", test)

        test = bp.where_intercepted('saturn')
        show_info("Saturn intercepted the old way", test)

        mask = bp.where_intercepted('saturn')
        test = bp.border_inside(mask)
        show_info("Saturn inside border", test)

        test = bp.border_outside(mask)
        show_info("Saturn outside border", test)

        test = bp.where_below(('ring_radius', 'saturn_main_rings'), 100000.)
        show_info("Ring area below 100,000 km", test)

        test = bp.border_below(('ring_radius', 'saturn_main_rings'), 100000.)
        show_info("Ring border below 100,000 km", test)

        test = bp.border_atop(('ring_radius', 'saturn_main_rings'), 100000.)
        show_info("Ring border atop 100,000 km", test)

        test = bp.border_atop(('ring_radius', 'saturn_main_rings'), 100000.)
        show_info("Ring border above 100,000 km", test)

        test = bp.evaluate(('border_atop', ('ring_radius',
                                            'saturn_main_rings'), 100000.))
        show_info("Ring border above 100,000 km via evaluate()", test)

        ########################
        # Testing ansa events and new notation

        test = bp.ring_radius('saturn_main_rings')
        show_info("Saturn main ring radius the old way (km)", test)

        test = bp.distance("saturn:ring")
        show_info("Saturn:ring radius (km)", test)

        test = bp.distance("saturn:ansa")
        show_info("Saturn:ansa distance (km)", test)

        test = bp.ansa_radius("saturn:ansa")
        show_info("Saturn:ansa radius (km)", test)

        test = bp.ansa_altitude("saturn:ansa")
        show_info("Saturn:ansa altitude (km)", test)

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
        # Testing ansa longitudes

        test = bp.ansa_longitude("saturn:ansa")
        show_info("Saturn ansa longitude wrt node (deg)", test * constants.DPR)

        test = bp.ansa_longitude("saturn:ansa", 'obs')
        show_info("Saturn ansa longitude wrt observer (deg)",
                                                           test * constants.DPR)

        test = bp.ansa_longitude("saturn:ansa", 'sun')
        show_info("Saturn ansa longitude wrt Sun (deg)", test * constants.DPR)

        ########################
        # Testing ring azimuth & elevation

        test = bp.ring_azimuth("saturn:ring")
        show_info("Saturn ring azimuth wrt observer(deg)", test * constants.DPR)

        compare = bp.ring_longitude("saturn:ring", 'obs')
        diff = test - compare
        show_info("Saturn ring azimuth wrt observer minus longitude (deg)",
                                                           diff * constants.DPR)

        test = bp.ring_azimuth("saturn:ring", reference='sun')
        show_info("Saturn ring azimuth wrt Sun (deg)", test * constants.DPR)

        compare = bp.ring_longitude("saturn:ring", 'sun')
        diff = test - compare
        show_info("Saturn ring azimuth wrt Sun minus longitude (deg)",
                                                           diff * constants.DPR)

        test = bp.ring_elevation("saturn:ring", reference='obs')
        show_info("Saturn ring elevation wrt observer (deg)",
                                                           test * constants.DPR)
        compare = bp.emission_angle("saturn:ring")
        diff = test + compare
        show_info("Saturn ring emission wrt observer plus emission (deg)",
                                                           diff * constants.DPR)

        test = bp.ring_elevation("saturn:ring", reference='sun')
        show_info("Saturn ring elevation wrt Sun (deg)", test * constants.DPR)

        compare = bp.incidence_angle("saturn:ring")
        diff = test + compare
        show_info("Saturn ring elevation wrt Sun plus incidence (deg)",
                                                           diff * constants.DPR)

        ########################
        # Ring scalars, other tests 5/14/12

        if UNITTEST_PRINT:
            print
            print "Sub-solar Saturn planetocentric latitude (deg) =",
            print bp.sub_solar_latitude('saturn').vals * constants.DPR

            print "Sub-solar Saturn planetographic latitude (deg) =",
            print bp.sub_solar_latitude('saturn',
                                        'graphic').vals * constants.DPR

            print "Sub-solar Saturn longitude wrt IAU (deg) =",
            print bp.sub_solar_longitude('saturn').vals * constants.DPR

            print "Solar distance to Saturn center (km) =",
            print bp.center_distance('saturn', 'sun').vals

            print
            print "Sub-observer Saturn planetocentric latitude (deg) =",
            print bp.sub_observer_latitude('saturn').vals * constants.DPR

            print "Sub-observer Saturn planetographic latitude (deg) =",
            print bp.sub_observer_latitude('saturn',
                                           'graphic').vals * constants.DPR

            print "Sub-observer Saturn longitude wrt IAU (deg) =",
            print bp.sub_observer_longitude('saturn').vals * constants.DPR

            print "Observer distance to Saturn center (km) =",
            print bp.center_distance('saturn').vals

            print
            print "Sub-solar ring latitude (deg) =",
            print bp.sub_solar_latitude("saturn:ring").vals * constants.DPR

            print "Sub-solar ring longitude wrt node (deg) =",
            print bp.sub_solar_longitude("saturn:ring").vals * constants.DPR

            print "Solar distance to ring center (km) =",
            print bp.center_distance("saturn:ring", 'sun').vals

            print "Sub-observer ring latitude (deg) =",
            print bp.sub_observer_latitude("saturn:ring").vals * constants.DPR

            print "Sub-observer ring longitude wrt node (deg) =",
            print bp.sub_observer_longitude("saturn:ring").vals * constants.DPR

            print "Observer distance to ring center (km) =",
            print bp.center_distance("saturn:ring").vals
            print

        test = bp.ring_longitude('saturn_main_rings', reference='sun')
        show_info("Ring longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='old-sun')
        show_info("Ring longitude wrt Sun (deg), old way", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='sha')
        show_info("Ring longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='old-sha')
        show_info("Ring longitude wrt SHA (deg), old way", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='obs')
        show_info("Ring longitude wrt observer (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='old-obs')
        show_info("Ring longitude wrt observer (deg), old way",
                                                    test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='oha')
        show_info("Ring longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.ring_longitude('saturn_main_rings', reference='old-oha')
        show_info("Ring longitude wrt OHA (deg), old way", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='sun')
        show_info("Ansa longitude wrt Sun (deg)", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='old-sun')
        show_info("Ansa longitude wrt Sun (deg), old way", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='sha')
        show_info("Ansa longitude wrt SHA (deg)", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='old-sha')
        show_info("Ansa longitude wrt SHA (deg), old way", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='obs')
        show_info("Ansa longitude wrt observer (deg)", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='old-obs')
        show_info("Ansa longitude wrt observer (deg), old way",
                                                    test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='oha')
        show_info("Ansa longitude wrt OHA (deg)", test * constants.DPR)

        test = bp.ansa_longitude("saturn_main_rings:ansa", reference='old-oha')
        show_info("Ansa longitude wrt OHA (deg), old way", test * constants.DPR)

        ########################
        # Limb and resolution, 6/6/12

#         test = bp.altitude("saturn:limb")
#         show_info("Limb altitude (km)", test)
# 
#         test = bp.longitude("saturn:limb")
#         show_info("Limb longitude (deg)", test * constants.DPR)
# 
#         test = bp.latitude("saturn:limb")
#         show_info("Limb latitude (deg)", test * constants.DPR)
# 
#         test = bp.longitude("saturn:limb", reference='obs', minimum=-180)
#         show_info("Limb longitude wrt observer, -180 (deg)",
#                                                         test * constants.DPR)
# 
#         test = bp.latitude("saturn:limb", lat_type='graphic')
#         show_info("Limb planetographic latitude (deg)", test * constants.DPR)
# 
#         test = bp.latitude("saturn:limb", lat_type='centric')
#         show_info("Limb planetocentric latitude (deg)", test * constants.DPR)
# 
#         test = bp.resolution("saturn:limb", 'u')
#         show_info("Limb resolution horizontal (km/pixel)", test)
# 
#         test = bp.resolution("saturn:limb", 'v')
#         show_info("Limb resolution vertical (km/pixel)", test)

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
                             (old_obs.pos, old_obs.vel),
                             old_obs.origin, old_obs.frame,
                             arr = old_obs.arr)

        bp.surface_events_w_derivs = {(): bp.obs_event}
        bp.surface_events = {(): bp.obs_event.without_derivs()}

        test = bp.distance('saturn')
        show_info("Range to Saturn, entirely masked (km)", test)

        test = bp.phase_angle('saturn')
        show_info("Phase angle at Saturn, entirely masked (deg)",
                                                test * constants.DPR)

        config.LOGGING.off()

        Path.reset_registry()
        Frame.reset_registry()
        iss.ISS.reset()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
