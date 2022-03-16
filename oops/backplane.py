################################################################################
# oops/backplane.py: Backplane class
#
# TBD...
#   position(self, event_key, axis='x', apparent=True, frame='j2000')
#       returns an arbitrary position vector in an arbitrary frame
#   velocity(self, event_key, axis='x', apparent=True, frame='j2000')
#       returns an arbitrary velocity vector in an arbitrary frame
#   projected_radius(self, event_key, radius='long')
#       returns the projected angular radius of the body along its long or
#       short axis.
################################################################################

from __future__ import print_function

import numpy as np
import os.path

from polymath import *

import oops.config    as config
import oops.constants as constants

from oops.surface_.surface     import Surface
from oops.surface_.ansa        import Ansa
from oops.surface_.limb        import Limb
from oops.surface_.ringplane   import RingPlane
from oops.surface_.nullsurface import NullSurface
from oops.path_.path           import Path, AliasPath
from oops.frame_.frame         import Frame
from oops.event                import Event
from oops.meshgrid             import Meshgrid
from oops.body                 import Body

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

class Backplane(object):
    """Backplane is a class that supports the generation and manipulation of
    sets of backplanes associated with a particular observation. It caches
    intermediate results to speed up calculations."""

    PACKRAT_ARGS = ['obs', 'meshgrid', 'time', 'inventory', 'inventory_border',
                    '+surface_events',
                    '+surface_events_w_derivs',
                    '+path_events',
                    '+gridless_events',
                    '+gridless_arrivals']

    def __init__(self, obs, meshgrid=None, time=None, inventory=None,
                            inventory_border=0):
        """The constructor.

        Input:
            obs         the Observation object with which this backplane is
                        associated.

            meshgrid    the optional meshgrid defining the sampling of the FOV.
                        Default is to sample the center of every pixel.

            time        an optional Scalar of times. The shape of this Scalar
                        will be broadcasted with the shape of the meshgrid.
                        Default is to sample the midtime of every pixel.

            inventory   True to keep an inventory of bodies in the field of
                        view and to keep track of their locations. This option
                        can speed up backplane calculations for bodies that
                        occupy a small fraction of the field of view. If a
                        dictionary is provided, this dictionary is used.

            inventory_border
                        number of pixels to extend the box surrounding each body
                        as determined by the inventory.
        """

        self.obs = obs

        if meshgrid is None:
            self.meshgrid = obs.meshgrid()
        else:
            self.meshgrid = meshgrid

        if time is None:
            self.time = obs.timegrid(self.meshgrid)
        else:
            self.time = Scalar.as_scalar(time)

        # For some cases, times are all equal. If so, collapse the times.
        dt = self.time - obs.midtime
        if abs(dt).max() < 1.e-3:   # simplifies cases with jitter in time tags
            self.time = Scalar(obs.midtime)

        # Intialize the inventory
        if type(inventory) == dict:
            self.inventory = inventory
        elif inventory:
            self.inventory = {}
        else:
            self.inventory = None

        self.inventory_border = inventory_border

        # Define events
        self.obs_event = obs.event_at_grid(self.meshgrid, self.time)
        self.obs_gridless_event = obs.gridless_event(self.meshgrid, self.time)

        self.shape = self.obs_event.shape

        # The surface_events dictionary comes in two versions, one with
        # derivatives and one without. Each dictionary is keyed by a tuple of
        # strings, where each string is the name of a body from which photons
        # depart. Each event is defined by a call to Surface.photon_to_event()
        # to the next. The last event is the arrival at the observation, which
        # is implied so it does not appear in the key. For example, ('SATURN',)
        # is the key for an event defining the departures of photons from the
        # surface of Saturn to the observer. Shadow calculations require a
        # pair of steps; for example, ('saturn','saturn_main_rings') is the key
        # for the event of intercept points on Saturn that fall in the path of
        # the photons arriving at Saturn's rings from the Sun.
        #
        # Note that body names in event keys are case-insensitive.

        self.surface_events_w_derivs = {(): self.obs_event}
        self.surface_events = {(): self.obs_event.wod}

        # The path_events dictionary holds photon departure events from paths.
        # All photons originate from the Sun so this name is implied. For
        # example, ('SATURN',) is the key for the event of a photon departing
        # the Sun such that it later arrives at the Saturn surface.

        self.path_events = {}

        # The gridless_events dictionary keeps track of photon events at the
        # origin point of each defined surface. It uses the same keys as the
        # surface_events dictionary.
        #
        # The observed counterpart to each gridless event is stored in
        # gridless_arrivals

        self.gridless_events   = {(): self.obs_gridless_event}
        self.gridless_arrivals = {}

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

        # Antimasks of surfaces, by body name

        self.antimasks = {}

        # Minimum radius of each ":RING" object is the radius of the central
        # body

        self.min_ring_radius = {}

    ############################################################################
    # Event manipulations
    #
    # Event keys are tuples (body_id, body_id, ...) where each body_id is a
    # a step along the path of a photon. The path always begins at the Sun and
    # and ends at the observer.
    ############################################################################

    def standardize_event_key(self, event_key):
        """Repair an event key to make it suitable for indexing a dictionary.

        Strings are converted to uppercase. A string gets turned into a tuple.
        """

        if type(event_key) == str:
            event_key = (event_key.upper(),)

        # elif isinstance(event_key, basestring):
        #     event_key = (str(event_key).upper(),)

        elif type(event_key) == tuple:
            items = []
            for item in event_key:
                if type(item) == str:
                    items.append(item.upper())
                else:
                    items.append(item)

            event_key = tuple(items)

        else:
            raise ValueError('illegal event key type: ' + str(type(event_key)))

        return event_key

    @staticmethod
    def standardize_backplane_key(backplane_key):
        """Repair a backplane key to make it suitable for indexing a dictionary.

        A string is turned into a tuple. Strings are converted to upper case. If
        the argument is a backplane already, the key is extracted from it.
        """

        if type(backplane_key) == str:
            return (backplane_key.upper(),)

        elif type(backplane_key) == tuple:
            return backplane_key

        elif isinstance(backplane_key, (Scalar,Boolean)):
            return backplane_key.key

        else:
            raise ValueError('illegal backplane key type: ' +
                              str(type(backplane_key)))

    @staticmethod
    def get_body_and_modifier(event_key):
        """Return a body object and modifier based on the given surface ID.

        The string is normally a registered body ID (case insensitive), but it
        can be modified with ':ansa', ':ring' or ':limb' to indicate an
        associated surface.
        """

        surface_id = event_key[0].upper()

        modifier = None
        if surface_id.endswith(':ANSA'):
            modifier = 'ANSA'
            body_id = surface_id[:-5]

        elif surface_id.endswith(':RING'):
            modifier = 'RING'
            body_id = surface_id[:-5]

        elif surface_id.endswith(':LIMB'):
            modifier = 'LIMB'
            body_id = surface_id[:-5]

        else:
            modifier = None
            body_id = surface_id

        return (Body.lookup(body_id), modifier)

    def get_surface(self, event_key):
        """Return a surface based on its ID."""

        (body, modifier) = Backplane.get_body_and_modifier(event_key)

        if modifier is None:
            return body.surface

        if modifier == 'RING':
            if event_key not in self.min_ring_radius:
                self.min_ring_radius[event_key] = body.surface.radii[0]

            if body.ring_body is not None:
                return body.ring_body.surface

            return RingPlane(body.path, body.ring_frame, gravity=body.gravity)

        if modifier == 'ANSA':
            if body.surface.COORDINATE_TYPE == 'polar':     # if it's a ring
                return Ansa.for_ringplane(body.surface)
            else:                                           # if it's a planet
                return Ansa(body.path, body.ring_frame)

        if modifier == 'LIMB':
            return Limb(body.surface, limits=(0.,np.inf))

    @staticmethod
    def get_path(path_id):
        """Return a path based on its ID."""

        return Body.lookup(path_id.upper()).path

    def get_antimask(self, event_key):
        """Prepare an antimask for a particular surface event."""

        # The basic meshgrid is unmasked
        if len(event_key) == 0:
            return True

        # Return from the antimask cache if present
        try:
            return self.antimasks[event_key]
        except KeyError:
            pass

        try:
            return self.antimasks[event_key[1:]]
        except KeyError:
            pass

        body_name = event_key[-1]

        # For a name with a colon, we're done
        if ':' in body_name:
            return True

        # Otherwise, use the inventory if available
        if self.inventory is not None:
            try:
                body_dict = self.inventory[body_name]

            # If it is not already in the inventory, try to make a new entry
            except KeyError:
              body_dict = None
              if self.obs.INVENTORY_IMPLEMENTED:
                try:
                  body_dict = self.obs.inventory([body_name],
                                                 return_type='full')[body_name]
                except KeyError:
                  pass

              self.inventory[body_name] = body_dict

            if body_dict is None:
                return False

            if not body_dict['inside']:
                return False

            u_min = body_dict['u_min'] - self.inventory_border
            u_max = body_dict['u_max'] + self.inventory_border
            v_min = body_dict['v_min'] - self.inventory_border
            v_max = body_dict['v_max'] + self.inventory_border

            antimask = np.ones(self.meshgrid.shape, dtype='bool')
            antimask[self.meshgrid.uv.values[...,0] <  u_min] = False
            antimask[self.meshgrid.uv.values[...,0] >= u_max] = False
            antimask[self.meshgrid.uv.values[...,1] <  v_min] = False
            antimask[self.meshgrid.uv.values[...,1] >= v_max] = False

            # Swap axes if necessary
            for c in self.obs.axes:
                if c[0] == 'v':
                    new_mask = antimask.swapaxes(0,1)
                    break
                if c[0] == 'u':
                    break

            return antimask

        return True

    def get_surface_event(self, event_key):
        """Return the photon departure event from a surface based on its key.
        """

        event_key = self.standardize_event_key(event_key)

        # If the event already exists, return it
        if event_key in self.surface_events:
            return self.surface_events[event_key]

        # The Sun is treated as a path, not a surface, unless it is listed last
        if event_key[0] == 'SUN' and len(event_key) > 1:
            return self.get_path_event(event_key)

        # Look up the photon's departure surface
        surface = self.get_surface(event_key)

        # Calculate derivatives for the first step from the observer, if allowed
        if len(event_key) == 1 and surface.intercept_DERIVS_ARE_IMPLEMENTED:
            try:
                event = self.get_surface_event_w_derivs(event_key)
                return self.surface_events[event_key]
            except NotImplementedError:
                pass

        # Look up the photon's destination
        dest = self.get_surface_event_with_arr(event_key[1:])

        # Define the antimask
        antimask = self.get_antimask(event_key)

        # Create the event and save it in the dictionary
        event = surface.photon_to_event(dest, antimask=antimask)[0]
        self.surface_events[event_key] = event
        if event_key not in self.antimasks:
            self.antimasks[event_key] = event.antimask

        # Save extra information in the event object
        event.insert_subfield('event_key', event_key)
        event.insert_subfield('surface', surface)

        body = Backplane.get_body_and_modifier(event_key)[0]
        event.insert_subfield('body', body)

        # Save the antimask
        if len(event_key) == 1 and event_key[0] not in self.antimasks:
            self.antimasks[event_key] = event.antimask

        return event

    def get_surface_event_w_derivs(self, event_key):
        """The photon departure event from a surface including derivatives.
        """

        event_key = self.standardize_event_key(event_key)

        # If the event already exists, return it
        if event_key in self.surface_events_w_derivs:
            return self.surface_events_w_derivs[event_key]

        # Look up the photon's departure surface
        surface = self.get_surface(event_key)

        # Look up the photons destination and prepare derivatives
        dest = self.get_surface_event_w_derivs(event_key[1:])
        dest = dest.with_time_derivs().with_los_derivs()

        # Define the antimask
        antimask = self.get_antimask(event_key)

        # Create the event and save it in the dictionary
        event = surface.photon_to_event(dest, derivs=True, antimask=antimask)[0]
        if event_key not in self.antimasks:
            self.antimasks[event_key] = event.antimask

        # Save extra information in the event object
        event.insert_subfield('event_key', event_key)
        event.insert_subfield('surface', surface)

        body = Backplane.get_body_and_modifier(event_key)[0]
        event.insert_subfield('body', body)

        self.surface_events_w_derivs[event_key] = event

        # Make a copy without derivs
        event_wo_derivs = event.wod
        event_wo_derivs.insert_subfield('event_key', event_key)
        event_wo_derivs.insert_subfield('surface', surface)
        event_wo_derivs.insert_subfield('body', body)
        self.surface_events[event_key] = event_wo_derivs

        # Also save into the dictionary keyed by the string name alone
        # Saves the bother of typing parentheses and a comma when it's not
        # really necessary
        if len(event_key) == 1:
            self.surface_events_w_derivs[event_key[0]] = event
            self.surface_events[event_key[0]] = event_wo_derivs

            # Also save the antimask
            self.antimasks[event_key[0]] = event.antimask

        return event

    def get_path_event(self, event_key):
        """Return the departure event from a specified path."""

        event_key = self.standardize_event_key(event_key)

        # If the event already exists, return it
        if event_key in self.path_events:
            return self.path_events[event_key]

        # Create the event
        dest = self.get_surface_event(event_key[1:])
        path = Backplane.get_path(event_key[0])

        event = path.photon_to_event(dest)[0]
        event.insert_subfield('event_key', event_key)

        self.path_events[event_key] = event

        # For a tuple of length 1, also register under the name string
        if len(event_key) == 1:
            self.path_events[event_key[0]] = event

        return event

    def get_surface_event_with_arr(self, event_key):
        """Return the specified event with arrival photons filled in."""

        event = self.get_surface_event(event_key)
        if event.arr is None and event.arr_ap is None:
            new_event = AliasPath('SUN').photon_to_event(event,
                                                    antimask=event.antimask)[1]
            new_event.insert_subfield('event_key', event_key)
            new_event.insert_subfield('surface', event.surface)
            new_event.insert_subfield('body', event.body)

            self.surface_events[event_key] = new_event
            return new_event

        return event

    def get_gridless_event(self, event_key):
        """Return the gridless event identifying a photon departure from a path.
        """

        event_key = self.standardize_event_key(event_key)

        # If the event already exists, return it
        if event_key in self.gridless_events:
            return self.gridless_events[event_key]

        # Create the event and save it in the dictionary
        dest = self.get_gridless_event(event_key[1:])
        surface = self.get_surface(event_key)

        (event, arrival) = surface.origin.photon_to_event(dest)
        event = event.wrt_frame(surface.frame)
        arrival = arrival.wrt_frame(self.obs.frame)

        # Save extra information in the event object
        body = Backplane.get_body_and_modifier(event_key)[0]

        event.insert_subfield('event_key', event_key)
        event.insert_subfield('surface', surface)
        event.insert_subfield('body', body)

        arrival.insert_subfield('event_key', event_key)
        arrival.insert_subfield('surface', surface)
        arrival.insert_subfield('body', body)

        self.gridless_events[event_key] = event
        self.gridless_arrivals[event_key] = arrival

        return event

    def get_gridless_event_with_arr(self, event_key):
        """Return the gridless event with the arrival photons been filled in.
        """

        event_key = self.standardize_event_key(event_key)

        event = self.get_gridless_event(event_key)
        if event.arr is not None:
            return event

        new_event = AliasPath('SUN').photon_to_event(event)[1]
        new_event.insert_subfield('event_key', event_key)
        new_event.insert_subfield('surface', event.surface)

        body = Backplane.get_body_and_modifier(event_key)[0]
        new_event.insert_subfield('body', body)

        self.gridless_events[event_key] = new_event
        return new_event

    def apply_mask_to_event(self, event_key, mask):
        """Apply the given mask to the event(s) associated with this event_key.
        """

        event_key = self.standardize_event_key(event_key)

        if event_key in self.surface_events:
            event = self.surface_events[event_key]
            new_event = event.mask_where(mask)
            new_event.insert_subfield('event_key', event_key)
            new_event.insert_subfield('surface', event.surface)
            new_event.insert_subfield('body', event.body)
            self.surface_events[event_key] = new_event

        if event_key in self.surface_events_w_derivs:
            event = self.surface_events_w_derivs[event_key]
            new_event = event.mask_where(mask)
            new_event.insert_subfield('event_key', event_key)
            new_event.insert_subfield('surface', event.surface)
            new_event.insert_subfield('body', event.body)
            self.surface_events_w_derivs[event_key] = new_event

    def mask_as_boolean(self, mask):
        """Converts a mask represented by a single boolean into a Boolean and
        makes sure Boolean us unmasked.
        """

        # Undefined values are treated as False
        if isinstance(mask, Qube):
            mask = mask.as_mask_where_nonzero()

        if isinstance(mask, np.ndarray): return Boolean(mask)

        # Convert a single value to a full array
        if mask:
            return Boolean(np.ones(self.shape, dtype='bool'))
        else:
            return Boolean(np.zeros(self.shape, dtype='bool'))

    def register_backplane(self, key, backplane, expand=True):
        """Insert this backplane into the dictionary.

        If expand is True and the backplane contains just a single value, the
        backplane is expanded to the overall shape."""

        if isinstance(backplane, (np.bool_, bool)):
            backplane = Boolean(bool(backplane))

        elif isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)

        # Under some circumstances a derived backplane can be a scalar
        if expand and backplane.shape == () and self.shape != ():
            if type(backplane) == Boolean:
                vals = np.empty(self.shape, dtype='bool')
                vals[...] = backplane.vals
                backplane = Boolean(vals, backplane.mask)
            else:
                vals = np.empty(self.shape, dtype='float')
                vals[...] = backplane.vals
                backplane = Scalar(vals, backplane.mask)

        # For reference, we add the key as an attribute of each backplane object
        backplane = backplane.wod
        backplane.key = key

        self.backplanes[key] = backplane

    def register_gridless_backplane(self, key, backplane):
        """Insert this backplane into the dictionary.

        Same as register_backplane() but without the expansion of a scalar
        value."""

        self.register_backplane(key, backplane, expand=False)

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

        event_key = self.standardize_event_key(event_key)

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

        event_key = self.standardize_event_key(event_key)

        key = ('declination', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

    def _fill_ra_dec(self, event_key, apparent, direction):
        """Fill internal backplanes of RA and dec."""

        assert direction in ('arr', 'dep')

        event = self.get_surface_event_with_arr(event_key)
        (ra, dec) = event.ra_and_dec(apparent=apparent, subfield=direction)

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

        event_key = self.standardize_event_key(event_key)
        key = ('celestial_north_angle', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

        temp_key = ('dlos_ddec', event_key)
        if temp_key not in self.backplanes:
            self._fill_dlos_dradec(event_key)

        dlos_ddec = self.backplanes[temp_key]
        duv_ddec = self.meshgrid.duv_dlos.chain(dlos_ddec)

        du_ddec_vals = duv_ddec.vals[...,0]
        dv_ddec_vals = duv_ddec.vals[...,1]
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

        event_key = self.standardize_event_key(event_key)
        key = ('celestial_east_angle', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

        temp_key = ('dlos_dra', event_key)
        if temp_key not in self.backplanes:
            self._fill_dlos_dradec(event_key)

        dlos_dra = self.backplanes[temp_key]
        duv_dra = self.meshgrid.duv_dlos.chain(dlos_dra)

        du_dra_vals = duv_dra.vals[...,0]
        dv_dra_vals = duv_dra.vals[...,1]
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

        # Convert to a column matrix and save
        dlos_dra  = Vector3(dlos_dradec.vals[...,0], ra.mask)
        dlos_ddec = Vector3(dlos_dradec.vals[...,1], ra.mask)

        self.register_backplane(('dlos_dra',  event_key), dlos_dra)
        self.register_backplane(('dlos_ddec', event_key), dlos_ddec)

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

        event_key = self.standardize_event_key(event_key)

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

        event_key = self.standardize_event_key(event_key)

        key = ('center_declination', event_key, apparent, direction)
        if key not in self.backplanes:
            self._fill_center_ra_dec(event_key, apparent, direction)

        return self.backplanes[key]

    def _fill_center_ra_dec(self, event_key, apparent, direction):
        """Internal method to fill in RA and dec for the center of a body."""

        assert direction in ('arr', 'dep')

        _ = self.get_gridless_event_with_arr(event_key)
        event = self.gridless_arrivals[event_key]

        (ra, dec) = event.ra_and_dec(apparent, subfield=direction)

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

        event_key = self.standardize_event_key(event_key)
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

        event_key = self.standardize_event_key(event_key)
        assert direction in ('dep', 'arr')

        key = ('light_time', event_key, direction)
        if key not in self.backplanes:
            if direction == 'arr':
                event = self.get_surface_event_with_arr(event_key)
                lt = event.arr_lt
            else:
                event = self.get_surface_event(event_key)
                lt = event.dep_lt

            self.register_backplane(key, abs(lt))

        return self.backplanes[key]

    def event_time(self, event_key):
        """Absolute time in seconds TDB when the photon intercepted the surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = self.standardize_event_key(event_key)

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

        event_key = self.standardize_event_key(event_key)
        assert axis in ('u','v')

        key = ('resolution', event_key, axis)
        if key not in self.backplanes:
            distance = self.distance(event_key)

            res = self.meshgrid.dlos_duv.swap_items(Pair)
            (u_resolution, v_resolution) = res.to_scalars()
            u_resolution = distance * u_resolution.join_items(Vector3).norm()
            v_resolution = distance * v_resolution.join_items(Vector3).norm()

            self.register_backplane(key[:-1] + ('u',), u_resolution)
            self.register_backplane(key[:-1] + ('v',), v_resolution)

        return self.backplanes[key]

    ############################################################################
    # Basic body, path intercept versions
    #   center_distance()   distance traveled by photon, km.
    #   center_light_time() elapsed time from photon departure to arrival, sec.
    #   center_resolution() resolution based on body center distance, km/pixel.
    ############################################################################

    def center_distance(self, event_key, direction='dep'):
        """Distance traveled by a photon between paths.

        Input:
            event_key       key defining the event at the body's path.
            direction       'arr' or 'sun' to return the distance traveled by an
                                           arriving photon;
                            'dep' or 'obs' to return the distance traveled by a
                                           departing photon.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('center_distance', event_key, direction)
        if key not in self.backplanes:
            lt = self.center_light_time(event_key, direction)
            self.register_gridless_backplane(key, lt * constants.C)

        return self.backplanes[key]

    def center_light_time(self, event_key, direction='dep'):
        """Light travel time in seconds from a path.

        Input:
            event_key       key defining the event at the body's path.
            direction       'arr' or 'sun' to return the distance traveled by an
                                           arriving photon;
                            'dep' or 'obs' to return the distance traveled by a
                                           departing photon.
        """

        event_key = self.standardize_event_key(event_key)
        assert direction in ('dep', 'arr', 'obs', 'sun')

        key = ('center_light_time', event_key, direction)
        if key not in self.backplanes:
            if direction in ('arr', 'sun'):
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

        event_key = self.standardize_event_key(event_key)

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

        event_key = self.standardize_event_key(event_key)
        assert axis in ('u','v')

        key = ('center_resolution', event_key, axis)
        if key not in self.backplanes:
            distance = self.center_distance(event_key)

            res = self.obs.fov.center_dlos_duv.swap_items(Pair)
            (u_resolution, v_resolution) = res.to_scalars()
            u_resolution = distance * u_resolution.join_items(Vector3).norm()
            v_resolution = distance * v_resolution.join_items(Vector3).norm()

            self.register_gridless_backplane(key[:-1] + ('u',), u_resolution)
            self.register_gridless_backplane(key[:-1] + ('v',), v_resolution)

        return self.backplanes[key]

    ############################################################################
    # Lighting geometry, surface intercept version
    #   incidence_angle()       incidence angle at surface, radians.
    #   emission_angle()        emission angle at surface, radians.
    #   lambert_law()           Lambert Law model for surface, cos(incidence).
    #   minnaert_law()          Minnaert Law model for surface.
    #   lommel_seeliger_law()   Lommel-Seeliger Law model for surface.
    ############################################################################

    def incidence_angle(self, event_key):
        """Incidence angle of the arriving photons at the local surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('incidence_angle', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            incidence = event.incidence_angle()

            # Ring incidence angles are always 0-90 degrees
            if event.surface.COORDINATE_TYPE == 'polar':

                # flip is True wherever incidence angle has to be changed
                flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
                self.register_backplane(('ring_flip', event_key), flip)

                # Now flip incidence angles where necessary
                incidence = Scalar.PI * flip + (1. - 2.*flip) * incidence

            self.register_backplane(key, incidence)

        return self.backplanes[key]

    def emission_angle(self, event_key):
        """Emission angle of the departing photons at the local surface.

        Input:
            event_key       key defining the surface event.
        """

        event_key = self.standardize_event_key(event_key)
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
                emission = Scalar.PI * flip + (1. - 2.*flip) * emission

            self.register_backplane(key, emission)

        return self.backplanes[key]

    def phase_angle(self, event_key):
        """Phase angle between the arriving and departing photons.

        Input:
            event_key       key defining the surface event.
        """

        event_key = self.standardize_event_key(event_key)
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

        event_key = self.standardize_event_key(event_key)
        key = ('scattering_angle', event_key)
        if key not in self.backplanes:
            self.register_backplane(key, Scalar.PI -
                                         self.phase_angle(event_key))

        return self.backplanes[key]

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

        event_key = self.standardize_event_key(event_key)
        key = ('center_incidence_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)

            # Sign on event.arr is negative because photon is incoming
            latitude = (event.neg_arr_ap.to_scalar(2) /
                        event.arr_ap.norm()).arcsin()
            incidence = Scalar.HALFPI - latitude

            # Ring incidence angles are always 0-90 degrees
            if event.surface.COORDINATE_TYPE == 'polar':

                # The flip is True wherever incidence angle has to be changed
                flip = Boolean.as_boolean(incidence > Scalar.HALFPI)
                self.register_gridless_backplane(('ring_center_flip',
                                                  event_key), flip)

                # Now flip incidence angles where necessary
                if flip.any():
                    incidence = Scalar.PI * flip + (1. - 2.*flip) * incidence

            self.register_gridless_backplane(key, incidence)

        return self.backplanes[key]

    def center_emission_angle(self, event_key):
        """Emission angle of the departing photons at the body's central path.

        This uses the z-axis of the body's frame to define the local normal.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('center_emission_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)

            latitude = (event.dep_ap.to_scalar(2) /
                        event.dep_ap.norm()).arcsin()
            emission = Scalar.HALFPI - latitude

            # Ring emission angles are always measured from the lit side normal
            if event.surface.COORDINATE_TYPE == 'polar':

                # Get the flip flag
                ignore = self.center_incidence_angle(event_key)
                flip = self.backplanes[('ring_center_flip', event_key)]

                # Flip emission angles where necessary
                if flip.any():
                    emission = Scalar.PI * flip + (1. - 2.*flip) * emission

            self.register_gridless_backplane(key, emission)

        return self.backplanes[key]

    def center_phase_angle(self, event_key):
        """Phase angle as measured at the body's central path.

        Input:
            event_key       key defining the event on the body's path.
        """

        event_key = self.standardize_event_key(event_key)
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

        event_key = self.standardize_event_key(event_key)
        key = ('center_scattering_angle', event_key)
        if key not in self.backplanes:
            angle = Scalar.PI - self.center_phase_angle(event_key)
            self.register_gridless_backplane(key, angle)

        return self.backplanes[key]

    ############################################################################
    # Body surface geometry, surface intercept versions
    #   longitude()
    #   latitude()
    ############################################################################

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
        # Note that longitudes default to eastward for right-handed coordinates
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
        if direction == 'west': lon = -lon

        # Re-define the minimum
        if minimum == 0:
            lon = lon % constants.TWOPI
        else:
            lon = (lon + constants.PI) % constants.TWOPI - Scalar.PI

        self.register_backplane(key, lon)
        return self.backplanes[key]

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

    def _sub_observer_longitude(self, event_key):
        """Sub-observer longitude. Used internally."""

        event_key = self.standardize_event_key(event_key)
        key = ('_sub_observer_longitude', event_key)

        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)
            dep_ap = event.apparent_dep()       # for ABERRATION=old or new
            lon = dep_ap.to_scalar(1).arctan2(dep_ap.to_scalar(0)) % \
                  constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

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

    def _sub_solar_longitude(self, event_key):
        """Sub-solar longitude. Used internally."""

        event_key = self.standardize_event_key(event_key)
        key = ('_sub_solar_longitude', event_key)

        if key not in self.backplanes:
            event = self.get_gridless_event_with_arr(event_key)
            neg_arr_ap = -event.apparent_arr()  # for ABERRATION=old or new
            lon = neg_arr_ap.to_scalar(1).arctan2(neg_arr_ap.to_scalar(0)) % \
                  constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

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

    ############################################################################
    # Surface geometry, path intercept versions
    #   sub_observer_longitude()
    #   sub_solar_longitude()
    #   sub_observer_latitude()
    #   sub_solar_latitude()
    ############################################################################

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
        if direction == 'west': lon = -lon

        # Re-define the minimum
        if minimum == 0:
            lon = lon % Scalar.TWOPI
        else:
            lon = (lon + Scalar.PI) % constants.TWOPI - Scalar.PI

        return lon

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

    ############################################################################

    def pole_clock_angle(self, event_key):
        """Projected pole vector on the sky, measured from north through wes.

        In other words, measured clockwise on the sky."""

        event_key = self.standardize_event_key(event_key)

        key = ('pole_clock_angle', event_key)
        if key not in self.backplanes:
            event = self.get_gridless_event(event_key)

            # Get the body frame's Z-axis in J2000 coordinates
            frame = Frame.J2000.wrt(event.frame)
            xform = frame.transform_at_time(event.time)
            pole_j2000 = xform.rotate(Vector3.ZAXIS)

            # Define the vector to the observer in the J2000 frame
            dep_j2000 = event.wrt_ssb().apparent_dep()

            # Construct a rotation matrix from J2000 to a frame in which the
            # Z-axis points along -dep and the J2000 pole is in the X-Z plane.
            # As it appears to the observer, the Z-axis points toward the body,
            # the X-axis points toward celestial north as projected on the sky,
            # and the Y-axis points toward celestial west (not east!).
            rotmat = Matrix3.twovec(-dep_j2000, 2, Vector3.ZAXIS, 0)

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

        event_key = self.standardize_event_key(event_key)

        key = ('pole_position_angle', event_key)
        if key not in self.backplanes:
            self.register_gridless_backplane(key,
                            Scalar.TWOPI - self.pole_clock_angle(event_key))

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

        event_key = self.standardize_event_key(event_key)
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

        event_key = self.standardize_event_key(event_key)
        key = ('coarsest_resolution', event_key)
        if key not in self.backplanes:
            self._fill_surface_resolution(event_key)

        return self.backplanes[key]

    def _fill_surface_resolution(self, event_key):
        """Internal method to fill in the surface resolution backplanes.
        """

        event_key = self.standardize_event_key(event_key)
        event = self.get_surface_event_w_derivs(event_key)

        dpos_duv = event.state.d_dlos.chain(self.meshgrid.dlos_duv)
        (minres, maxres) = Surface.resolution(dpos_duv)

        self.register_backplane(('finest_resolution', event_key),
                                minres)
        self.register_backplane(('coarsest_resolution', event_key),
                                maxres)

    ############################################################################
    # Limb geometry
    #   limb_altitude(limit)
    #   Note: longitude() and latitude() work for limb coordinates,
    ############################################################################

    def altitude(self, event_key, limit=None):
        """Deprecated name for limb_altitude()."""

        return self.limb_altitude(event_key)

    def limb_altitude(self, event_key, limit=None, remask=False):
        """Elevation of a limb point above the body's surface.

        Input:
            event_key       key defining the ring surface event.
            limit           upper limit to altitude in km. Higher altitudes are
                            masked.
            remask          if True, the limit will be applied to the default
                            event, so that all backplanes generated from this
                            event_key will have the same upper limit. This can
                            only be applied the first time this event_key is
                            used.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('limb_altitude', event_key, limit)
        if key in self.backplanes:
            return self.backplanes[key]

        self._fill_limb_intercepts(event_key, limit, remask)
        return self.backplanes[key]

    def _fill_limb_intercepts(self, event_key, limit=None, remask=False):
        """Internal method to fill in the limb intercept geometry backplanes.

        Input:
            event_key       key defining the ring surface event.
            limit           upper limit to altitude in km. Higher altitudes are
                            masked.
            remask          if True, the limit will be applied to the default
                            event, so that all backplanes generated from this
                            event_key will share the same limit. This can
                            only be applied the first time this event_key is
                            used.
        """

        # Don't allow remask if the backplane was already generated
        if limit is None: remask = False

        if remask and event_key in self.surface_events:
            raise ValueError('remask disallowed for pre-existing ' +
                             'limb event key ' + str(event_key))

        # Get the limb intercept coordinates
        event = self.get_surface_event(event_key)
        if event.surface.COORDINATE_TYPE != 'limb':
            raise ValueError('limb intercepts require a "limb" surface type')

        # Limit the event if necessary
        if remask:

            # Apply the upper limit to the event
            altitude = event.coord3
            self.apply_mask_to_event(event_key, altitude > limit)
            event = self.get_surface_event(event_key)

        # Register the default backplanes
        self.register_backplane(('longitude', event_key, 'iau', 'east', 0,
                                 'squashed'), event.coord1)
        self.register_backplane(('latitude', event_key, 'squashed'),
                                event.coord2)
        self.register_backplane(('limb_altitude', event_key, None),
                                event.coord3)

        # Apply a mask just to this backplane if necessary
        if limit is not None:
            altitude = event.coord3.mask_where_gt(limit)
            self.register_backplane(('limb_altitude', event_key, limit),
                                    altitude)

    ############################################################################
    # Ring plane geometry, surface intercept version
    #   ring_radius()
    #   ring_longitude()
    #   ring_azimuth()
    #   ring_elevation()
    ############################################################################

    def ring_radius(self, event_key, rmin=None, rmax=None, remask=False):
        """Radius of the ring intercept point in the observation.

        Input:
            event_key       key defining the ring surface event.
            rmin            minimum radius in km; None to allow it to be defined
                            by the event_key.
            rmax            maximum radius in km; None to allow it to be defined
                            by the event_key.
            remask          if True, the rmin and rmax values will be applied to
                            the default event, so that all backplanes generated
                            from this event_key will have the same limits. This
                            option can only be applied the first time this
                            event_key is used.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('ring_radius', event_key, rmin, rmax)
        if key in self.backplanes:
            return self.backplanes[key]

        default_key = ('ring_radius', event_key, None, None)
        if default_key not in self.backplanes:
            self._fill_ring_intercepts(event_key, rmin, rmax, remask)

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

    def ring_longitude(self, event_key, reference='node', rmin=None, rmax=None,
                             remask=False):
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
            rmin            minimum radius in km; None to allow it to be defined
                            by the event_key.
            rmax            maximum radius in km; None to allow it to be defined
                            by the event_key.
            remask          if True, the rmin and rmax values will be applied to
                            the default event, so that all backplanes generated
                            from this event_key will have the same limits. This
                            option can only be applied the first time this
                            event_key is used.
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
            self._fill_ring_intercepts(event_key, rmin, rmax, remask)

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
                ref_lon = self._sub_solar_longitude(event_key) - np.pi
            elif reference == 'obs':
                ref_lon = self._sub_observer_longitude(event_key)
            elif reference == 'oha':
                ref_lon = self._sub_observer_longitude(event_key) - np.pi

            lon = (self.backplanes[default_key] - ref_lon) % constants.TWOPI

            self.register_backplane(reflon_key, lon)

        # Apply the radial mask if necessary
        if rmin is None and rmax is None:
            return self.backplanes[key]

        mask = self.ring_radius(event_key, rmin, rmax).mask
        lon = lon.mask_where(mask)
        self.register_backplane(key, lon)

        return lon

    def radial_mode(self, backplane_key,
                          cycles, epoch, amp, peri0, speed, a0=0., dperi_da=0.,
                          reference='node'):
        """Radius shift based on a particular ring mode.

        Input:
            backplane_key   key defining a ring_radius backplane, possibly with
                            other radial modes.
            cycles          the number of radial oscillations in 360 degrees of
                            longitude.
            epoch           the time (seconds TDB) at which the mode parameters
                            apply.
            amp             radial amplitude of the mode in km.
            peri0           a longitude (radians) at epoch where the mode is at
                            its radial minimum at semimajor axis a0. For cycles
                            == 0, it is the phase at epoch, where a phase of 0
                            corresponds to the minimum ring radius, with every
                            particle at pericenter.
            speed           local pattern speed in radians per second, as scaled
                            by the number of cycles.
            a0              the reference semimajor axis, used for slopes
            dperi_da        the rate of change of pericenter with semimajor
                            axis, measured at semimajor axis a0 in radians/km.
            reference       the reference longitude used to describe the mode;
                            same options as for ring_longitude
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
            lon = (-frame.node_at_time(event.time)) % constants.TWOPI

            self.register_gridless_backplane(key, lon)

        return self.backplanes[key]

    def ring_azimuth(self, event_key, reference='obs'):
        """Angle from a reference direction to the local radial direction.

        The angle is measured in the prograde direction from a reference
        direction to the local radial, as measured at the ring intercept point
        and projected into the ring plane. This value is 90 degrees at the left
        ansa and 270 degrees at the right ansa."

        The reference direction can be 'obs' for the apparent departing
        direction of the photon, or 'sun' for the (negative) apparent direction
        of the arriving photon.

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
        az = (rad_angle - ref_angle) % constants.TWOPI
        self.register_backplane(key, az)

        return self.backplanes[key]

    def ring_elevation(self, event_key, reference='obs', signed=True):
        """Angle from the ring plane to the photon direction.

        Evaluated at the ring intercept point. The angle is positive on the side
        of the ring plane where rotation is prograde; negative on the opposite
        side.

        The reference direction can be 'obs' for the apparent departing
        direction of the photon, or 'sun' for the (negative) apparent direction
        of the arriving photon.

        Input:
            event_key       key defining the ring surface event.
            reference       'obs' or 'sun'; see discussion above.
            signed          True for elevations on the retrograde side of the
                            rings to be negative; False for all angles to be
                            non-negative.
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

    def _fill_ring_intercepts(self, event_key, rmin=None, rmax=None,
                                    remask=False):
        """Internal method to fill in the ring intercept geometry backplanes.

        Input:
            event_key       key defining the ring surface event.
            rmax            lower limit to the ring radius in km. Smaller radii
                            are masked. Note that radii inside the planet are
                            always masked.
            rmax            upper limit to the ring radius in km. Larger radii
                            are masked.
            remask          if True, the limits will be applied to the default
                            event, so that all backplanes generated from this
                            event_key will share the same limit. This can
                            only be applied the first time this event_key is
                            used.
        """

        # Don't allow remask if the backplane was already generated
        if rmin is None and rmax is None: remask = False

        if remask and event_key in self.surface_events:
            raise ValueError('remask disallowed for pre-existing ' +
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
        if remask:

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

    ############################################################################
    # Ring plane geometry, path intercept versions
    #   ring_sub_observer_longitude()
    #   ring_sub_solar_longitude()
    #   ring_center_incidence_angle()
    #   ring_center_emission_angle()
    ############################################################################

    def ring_sub_observer_longitude(self, event_key, reference='node'):
        """Sub-observer longitude in the ring plane.

        Input:
            event_key       key defining the event on the center of the ring's
                            path.
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

    def ring_sub_solar_longitude(self, event_key, reference='node'):
        """Sub-solar longitude in the ring plane.

        Input:
            event_key       key defining the event on the center of the ring's
                            path.
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

    def ring_center_incidence_angle(self, event_key, pole='sunward'):
        """Incidence angle of the arriving photons at the ring system center.

        By default, angles are measured from the sunward pole and should always
        be <= pi/2. However, calculations for values relative to the IAU-defined
        north pole and relative to the prograde pole are also supported.

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

    ############################################################################
    # Ring plane geometry, surface intercept only
    #   ring_radial_resolution()
    #   ring_angular_resolution()
    #   ring_gradient_angle()
    ############################################################################

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
        drad_duv = rad.d_dlos.chain(self.meshgrid.dlos_duv)
        res = drad_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

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
        dlon_duv = lon.d_dlos.chain(self.meshgrid.dlos_duv)
        res = dlon_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

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
        drad_duv = rad.d_dlos.chain(self.meshgrid.dlos_duv)
        (drad_du, drad_dv) = drad_duv.join_items(Pair).to_scalars()

        clock = drad_dv.arctan2(drad_du)
        self.register_backplane(key, clock)

        return self.backplanes[key]

    ############################################################################
    # Ring shadow calculation
    #   ring_shadow_radius()
    #   ring_radius_in_front()
    ############################################################################

    def ring_shadow_radius(self, event_key, ring_body):
        """Radius in the ring plane that casts a shadow at each point on this
        body."""

        event_key = self.standardize_event_key(event_key)
        ring_body = self.standardize_event_key(ring_body)[0]

        key = ('ring_shadow_radius', event_key, ring_body)
        if key not in self.backplanes:
            event = self.get_surface_event_with_arr(event_key)
            ring_event = self.get_surface_event((ring_body,) + event_key)
            radius = ring_event.coord1
            self.register_backplane(key, radius)

        return self.backplanes[key]

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

    ############################################################################
    # Ring ansa geometry, surface intercept only
    #   ansa_radius()
    #   ansa_altitude()
    #   ansa_longitude()
    ############################################################################

    def ansa_radius(self, event_key, radius_type='positive', rmax=None,
                          remask=False):
        """Radius of the ring ansa intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
            radius_type     'right' for radii increasing rightward when prograde
                                    rotation pole is 'up';
                            'left' for the opposite of 'right';
                            'positive' for all radii using positive values.
            rmax            maximum absolute value of the radius in km; None to
                            allow it to be defined by the event_key.
            remask          if True, the rmax value will be applied to the
                            default event, so that all backplanes generated
                            from this event_key will have the same limits. This
                            option can only be applied the first time this
                            event_key is used.
        """

        # Look up under the desired radius type
        event_key = self.standardize_event_key(event_key)
        key0 = ('ansa_radius', event_key)
        key = key0 + (radius_type, rmax)
        if key in self.backplanes:
            return self.backplanes[key]

        # If not found, look up the default 'right'
        assert radius_type in ('right', 'left', 'positive')

        key_default = key0 + ('right', None)
        if key_default not in self.backplanes:
            self._fill_ansa_intercepts(event_key, rmax, remask)

        key_right = key0 + ('right', rmax)
        backplane = self.backplanes[key_right]

        if radius_type == 'left':
            self.register_backplane(key, -backplane)
        elif radius_type == 'positive':
            self.register_backplane(key, backplane.abs())

        return self.backplanes[key]

    def ansa_altitude(self, event_key):
        """Elevation of the ring ansa intercept point in the image.

        Input:
            event_key       key defining the ring surface event.
        """

        event_key = self.standardize_event_key(event_key)
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
        """

        event_key = self.standardize_event_key(event_key)
        assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

        # Look up under the desired reference
        key0 = ('ansa_longitude', event_key)
        key = key0 + (reference,)
        if key in self.backplanes:
            return self.backplanes[key]

        # If it is not found with reference J2000, fill in those backplanes
        key_node = key0 + ('node',)
        if key_node not in self.backplanes:
            self._fill_ansa_longitudes(event_key)

        # Now apply the reference longitude
        if reference == 'node':
            return self.backplanes[key]

        if reference == 'aries':
            ref_lon = self._aries_ring_longitude(event_key)
        elif reference == 'sun':
            ref_lon = self._sub_solar_ansa_longitude(event_key)
        elif reference == 'sha':
            ref_lon = self._sub_solar_ansa_longitude(event_key) - Scalar.PI
        elif reference == 'obs':
            ref_lon = self._sub_observer_ansa_longitude(event_key)
        elif reference == 'oha':
            ref_lon = self._sub_observer_ansa_longitude(event_key) - Scalar.PI

        lon = (self.backplanes[key_node] - ref_lon) % constants.TWOPI
        self.register_backplane(key, lon)

        return self.backplanes[key]

    def _fill_ansa_intercepts(self, event_key, rmax=None, remask=False):
        """Internal method to fill in the ansa intercept geometry backplanes.

        Input:
            rmax            maximum absolute value of the radius in km; None to
                            allow it to be defined by the event_key.
            remask          if True, the rmax value will be applied to the
                            default event, so that all backplanes generated
                            from this event_key will have the same limits. This
                            option can only be applied the first time this
                            event_key is used.
        """

        # Don't allow remask if the backplane was already generated
        if rmax is None: remask = False

        if remask and event_key in self.surface_events:
            raise ValueError('remask disallowed for pre-existing ' +
                             'ansa event key ' + str(event_key))

        # Get the ansa intercept coordinates
        event = self.get_surface_event(event_key)
        if event.surface.COORDINATE_TYPE != 'cylindrical':
            raise ValueError('ansa intercepts require a "cylindrical" ' +
                             'surface type')

        # Limit the event if necessary
        if remask:

            # Apply the upper limit to the event
            radius = event.coord1.abs()
            self.apply_mask_to_event(event_key, radius > rmax)
            event = self.get_surface_event(event_key)

        # Register the default backplanes
        self.register_backplane(('ansa_radius', event_key, 'right', None),
                                event.coord1)
        self.register_backplane(('ansa_altitude', event_key),
                                event.coord2)

        # Apply a mask just to these backplanes if necessary
        if rmax is not None:
            mask = (event.coord1 > rmax)
            self.register_backplane(('ansa_radius', event_key, 'right', rmax),
                                    event.coord1.mask_where(mask))
            self.register_backplane(('ansa_altitude', event_key, rmax),
                                    event.coord2.mask_where(mask))

    def _fill_ansa_longitudes(self, event_key):
        """Internal method to fill in the ansa intercept longitude backplane.
        """

        # Get the ansa intercept event
        event_key = self.standardize_event_key(event_key)
        event = self.get_surface_event(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        # Get the longitude in the associated ring plane
        lon = event.surface.ringplane.coords_from_vector3(event.pos, axes=2)[1]
        self.register_backplane(('ansa_longitude', event_key, 'node'),
                                lon)

    def _sub_observer_ansa_longitude(self, event_key):
        """Sub-observer longitude evaluated at the ansa intercept time.

        Used only internally. DEPRECATED.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('_sub_observer_ansa_longitude', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

        # At each intercept time, determine the outgoing direction to the
        # observer from the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, Vector3.ZERO,
                                         event.origin, event.frame)
        center_event = self.obs_event.origin.photon_from_event(center_event)[1]

        surface = self.get_surface(event_key).ringplane
        (r,lon) = surface.coords_from_vector3(center_event.apparent_dep(),
                                              axes=2)

        self.register_gridless_backplane(key, lon)
        return self.backplanes[key]

    def _sub_solar_ansa_longitude(self, event_key):
        """Sub-solar longitude evaluated at the ansa intercept time.

        Used only internally. DEPRECATED.
        """

        event_key = self.standardize_event_key(event_key)
        key = ('_sub_solar_ansa_longitude', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

        # At each intercept time, determine the incoming direction from the Sun
        # to the center of the planet
        event = self.get_surface_event(event_key)
        center_event = Event(event.time, Vector3.ZERO,
                                         event.origin, event.frame)
        center_event = AliasPath('SUN').photon_to_event(center_event)[1]

        surface = self.get_surface(event_key).ringplane
        (r,lon) = surface.coords_from_vector3(-center_event.apparent_arr(),
                                              axes=2)

        self.register_gridless_backplane(key, lon)
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

        event_key = self.standardize_event_key(event_key)
        key = ('ansa_radial_resolution', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

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

        event_key = self.standardize_event_key(event_key)
        key = ('ansa_vertical_resolution', event_key)
        if key in self.backplanes:
            return self.backplanes[key]

        event = self.get_surface_event_w_derivs(event_key)
        assert event.surface.COORDINATE_TYPE == 'cylindrical'

        z = event.coord2
        dz_duv = z.d_dlos.chain(self.meshgrid.dlos_duv)
        res = dz_duv.join_items(Pair).norm()

        self.register_backplane(key, res)
        return self.backplanes[key]

    ############################################################################
    # Orbit geometry, path intercept version
    #   orbit_longitude()
    ############################################################################

    def orbit_longitude(self, event_key, reference='obs', planet=None):
        """Longitude on an orbit path relative to the central planet.

        Input:
            event_key       key defining the event on the orbit path.
            reference       defines the location of zero longitude.
                            'aries' for the First point of Aries;
                            'node'  for the J2000 ascending node;
                            'obs'   for the sub-observer longitude;
                            'sun'   for the sub-solar longitude;
                            'oha'   for the anti-observer longitude;
                            'sha'   for the anti-solar longitude, returning the
                                    solar hour angle.
            planet          ID of the body at the center of the orbit; None for
                            the default, which is the parent of the targeted
                            body.
        """

        assert reference in {'aries', 'node', 'obs', 'oha', 'sun', 'sha'}

        # Determine/validate the planet
        event_key = self.standardize_event_key(event_key)
        if planet is None:
            (body,_) = self.get_body_and_modifier(event_key)
            planet = body.parent.name

        # Get the event
        if reference in {'sun', 'sha'}:
            orbit_event = self.get_gridless_event_with_arr(event_key)
        else:
            orbit_event = self.get_gridless_event(event_key)

        # Look up under the reference
        key0 = ('orbit_longitude', event_key)
        key = key0 + (reference, planet)
        if key in self.backplanes:
            return self.backplanes[key]

        planet_body = Body.lookup(planet)
        planet_event = planet_body.path.event_at_time(orbit_event.time)
        planet_event = planet_event.wrt_frame(Frame.J2000)
        orbit_event = orbit_event.wrt(planet_body.path, Frame.J2000)

        # Locate reference vector in the J2000 frame
        if reference == 'obs':
            reference_dir = orbit_event.dep_ap.wod
        elif reference == 'oha':
            reference_dir = -orbit_event.dep_ap.wod
        elif reference == 'sun':
            reference_dir = orbit_event.neg_arr_ap.wod
        elif reference == 'sha':
            reference_dir = orbit_event.arr_ap.wod
        elif reference == 'aries':
            reference_dir = Vector3.XAXIS
        else:           # 'node'
            pole = orbit_event.pos.cross(orbit_event.vel)
            reference_dir = Vector3.ZAXIS.cross(pole)

        # This matrix rotates J2000 coordinates to a frame in which the Z-axis
        # is the orbit pole and the moon is instantaneously at the X-axis.
        matrix = Matrix3.twovec(orbit_event.pos, 0, orbit_event.vel, 1)

        # Locate the reference direction in this frame
        reference_wrt_orbit = matrix.rotate(reference_dir)
        x = reference_wrt_orbit.to_scalar(0)
        y = reference_wrt_orbit.to_scalar(1)
        ref_lon_wrt_orbit = y.arctan2(x,recursive=False)

        # Convert to an orbit with respect to the reference direction
        orbit_lon_wrt_ref = (-ref_lon_wrt_orbit) % constants.TWOPI
        self.register_gridless_backplane(key, orbit_lon_wrt_ref)
        return orbit_lon_wrt_ref

    ############################################################################
    # Masks
    ############################################################################

    def where_intercepted(self, event_key):
        """A mask where the surface was intercepted.
        """

        event_key  = self.standardize_event_key(event_key)
        key = ('where_intercepted', event_key)
        if key not in self.backplanes:
            event = self.get_surface_event(event_key)
            mask = self.mask_as_boolean(event.antimask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_inside_shadow(self, event_key, shadow_body):
        """A mask where the surface is in the shadow of a second body."""

        event_key = self.standardize_event_key(event_key)
        shadow_body = self.standardize_event_key(shadow_body)

        key = ('where_inside_shadow', event_key, shadow_body[0])
        if key not in self.backplanes:
            shadow_event = self.get_surface_event(shadow_body + event_key)
            mask = self.mask_as_boolean(shadow_event.antimask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_outside_shadow(self, event_key, shadow_body):
        """A mask where the surface is outside the shadow of a second body."""

        event_key  = self.standardize_event_key(event_key)
        shadow_body = self.standardize_event_key(shadow_body)

        key = ('where_outside_shadow', event_key, shadow_body[0])
        if key not in self.backplanes:
            mask = self.where_intercepted(event_key)
            mask = mask & ~self.where_inside_shadow(event_key, shadow_body)
            mask = self.mask_as_boolean(mask)
            self.register_backplane(key, mask)

        return self.backplanes[key]

    def where_in_front(self, event_key, back_body):
        """A mask where the first surface is in not obscured by the second
        surface.

        This is where the back_body is either further away than the front body
        or not intercepted at all."""

        event_key = self.standardize_event_key(event_key)
        back_body  = self.standardize_event_key(back_body)

        key = ('where_in_front', event_key, back_body[0])
        if key not in self.backplanes:
            front_distance = self.distance(event_key)
            back_distance  = self.distance(back_body)
            boolean = front_distance.values < back_distance.values
            boolean |= back_distance.mask
            boolean &= front_distance.antimask
            self.register_backplane(key, self.mask_as_boolean(boolean))

        return self.backplanes[key]

    def where_in_back(self, event_key, front_body):
        """A mask where the first surface is behind (obscured by) the second
        surface."""

        event_key = self.standardize_event_key(event_key)
        front_body = self.standardize_event_key(front_body)

        key = ('where_in_back', event_key, front_body[0])
        if key not in self.backplanes:
            boolean = (self.distance(event_key) > self.distance(front_body))
            self.register_backplane(key, self.mask_as_boolean(boolean))

        return self.backplanes[key]

    def where_sunward(self, event_key):
        """A mask where the surface of a body is facing toward the Sun."""

        event_key = self.standardize_event_key(event_key)
        key = ('where_sunward',) + event_key
        if key not in self.backplanes:
            incidence = self.incidence_angle(event_key)
            boolean = (incidence <= constants.HALFPI)
            self.register_backplane(key, self.mask_as_boolean(boolean))

        return self.backplanes[key]

    def where_antisunward(self, event_key):
        """A mask where the surface of a body is facing away fron the Sun."""

        event_key = self.standardize_event_key(event_key)
        key = ('where_antisunward',) + event_key
        if key not in self.backplanes:
            incidence = self.incidence_angle(event_key)
            boolean = (incidence > constants.HALFPI)
            self.register_backplane(key, self.mask_as_boolean(boolean))

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
            boolean = (backplane <= value)
            self.register_backplane(key, self.mask_as_boolean(boolean))

        return self.backplanes[key]

    def where_above(self, backplane_key, value):
        """A mask where the backplane is >= the specified value."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_above', backplane_key, value)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            boolean = (backplane >= value)
            self.register_backplane(key, self.mask_as_boolean(boolean))

        return self.backplanes[key]

    def where_between(self, backplane_key, low, high):
        """A mask where the backplane is between the given values, inclusive."""

        backplane_key = Backplane.standardize_backplane_key(backplane_key)
        key = ('where_between', backplane_key, low, high)
        if key not in self.backplanes:
            backplane = self.evaluate(backplane_key)
            boolean = (backplane >= low) & (backplane <= high)
            self.register_backplane(key, self.mask_as_boolean(boolean))

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

            self.register_backplane(key, Boolean(border & backplane.antimask))

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
            'scattering_angle', 'lambert_law', 'minnaert_law',
        'center_incidence_angle', 'center_emission_angle', 'center_phase_angle',
            'center_scattering_angle',

        'longitude', 'latitude',
        'sub_observer_longitude', 'sub_observer_latitude',
        'sub_solar_longitude', 'sub_solar_latitude',
        'pole_clock_angle', 'pole_position_angle',
        'finest_resolution', 'coarsest_resolution',
        'limb_altitude', 'altitude',

        'ring_radius', 'ring_longitude', 'radial_mode',
        'ring_azimuth', 'ring_elevation',
        'ring_incidence_angle', 'ring_emission_angle',
        'ring_sub_observer_longitude', 'ring_sub_solar_longitude',
        'ring_center_incidence_angle', 'ring_center_emission_angle',
        'ring_radial_resolution', 'ring_angular_resolution',
        'ring_gradient_angle',
        'ring_shadow_radius', 'ring_radius_in_front',

        'ansa_radius', 'ansa_altitude', 'ansa_longitude',
        'ansa_radial_resolution', 'ansa_vertical_resolution',

        'orbit_longitude',

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

        if type(backplane_key) == str: backplane_key = (backplane_key,)

        func = backplane_key[0]
        if func not in Backplane.CALLABLES:
            raise ValueError('unrecognized backplane function: ' + func)

        return Backplane.__dict__[func].__call__(self, *backplane_key[1:])

################################################################################
# BACKPLANE LOGGER FOR TESTING
################################################################################

def exercise_backplanes(filespec, printing, logging, saving, undersample=16,
                                  use_inventory=False, inventory_border=2):
    """Generates info from every backplane."""

    import numbers
    import oops.inst.cassini.iss as iss
    from PIL import Image

    def save_image(image, filename, lo=None, hi=None, zoom=1.):
        """Save an image file of a 2-D array.

        Input:
            image       a 2-D array.
            filename    the name of the output file, which should end with the
                        type, e.g., '.png' or '.jpg'
            lo          the array value to map to black; if None, then the
                        minimum value in the array is used.
            hi          the array value to map to white; if None, then the
                        maximum value in the array is used.
            zoom        the scale factor by which to enlarge the image, default
                        1.
        """

        image = np.asfarray(image)

        if lo is None:
            lo = image.min()

        if hi is None:
            hi = image.max()

        if zoom != 1:
            image = zoom_image(image, zoom, order=1)

        if hi == lo:
            bytes = np.zeros(image.shape, dtype='uint8')
        else:
            scaled = (image[::-1] - lo) / float(hi - lo)
            bytes = (256.*scaled).clip(0,255).astype('uint8')

        im = Image.frombytes('L', (bytes.shape[1], bytes.shape[0]), bytes)
        im.save(filename)

    def show_info(title, array):
        """Internal method to print summary information and display images as
        desired."""

        if not printing and not saving: return

        if printing: print(title)

        # Scalar summary
        if isinstance(array, numbers.Number):
            print('  ', array)

        # Mask summary
        elif type(array.vals) == bool or \
                (isinstance(array.vals, np.ndarray) and \
                 array.vals.dtype == np.dtype('bool')):
            count = np.sum(array.vals)
            total = np.size(array.vals)
            percent = int(count / float(total) * 100. + 0.5)
            print('  ', (count, total-count),
                        (percent, 100-percent), '(True, False pixels)')
            minval = 0.
            maxval = 1.

        # Unmasked backplane summary
        elif array.mask is False:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print('  ', minval)
            else:
                print('  ', (minval, maxval), '(min, max)')

        # Masked backplane summary
        else:
            print('  ', (array.min().as_builtin(),
                         array.max().as_builtin()), '(masked min, max)')
            total = np.size(array.mask)
            masked = np.sum(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print('  ', (masked, total-masked),
                        (percent, 100-percent), '(masked, unmasked pixels)')

            if total == masked:
                minval = np.min(array.vals)
                maxval = np.max(array.vals)
            else:
                minval = array.min().as_builtin()
                maxval = array.max().as_builtin()

        if saving and array.shape != ():
            if minval == maxval:
                maxval += 1.

            image = array.vals.copy()
            image[array.mask] = minval - 0.05 * (maxval - minval)

            filename = 'backplane-' + title + '.png'
            filename = filename.replace(':','_')
            filename = filename.replace('/','_')
            filename = filename.replace(' ','_')
            filename = filename.replace('(','_')
            filename = filename.replace(')','_')
            filename = filename.replace('[','_')
            filename = filename.replace(']','_')
            filename = filename.replace('&','_')
            filename = filename.replace(',','_')
            filename = filename.replace('-','_')
            filename = filename.replace('__','_')
            filename = filename.replace('__','_')
            filename = filename.replace('_.','.')
            save_image(image, filename)

    if printing and logging: config.LOGGING.on('        ')

    if printing: print()

    snap = iss.from_file(filespec)
    meshgrid = Meshgrid.for_fov(snap.fov, undersample=undersample, swap=True)

    if use_inventory:
        bp = Backplane(snap, meshgrid, inventory={})
    else:
        bp = Backplane(snap, meshgrid, inventory=None)

    ########################

    if printing: print('\n********* right ascension')

    test = bp.right_ascension(apparent=False)
    show_info('Right ascension (deg, actual)', test * constants.DPR)

    test = bp.right_ascension(apparent=True)
    show_info('Right ascension (deg, apparent)', test * constants.DPR)

    test = bp.center_right_ascension('saturn', apparent=False)
    show_info('Right ascension of Saturn (deg, actual)', test * constants.DPR)

    test = bp.center_right_ascension('saturn', apparent=True)
    show_info('Right ascension of Saturn (deg, apparent)', test * constants.DPR)

    test = bp.center_right_ascension('epimetheus', apparent=False)
    show_info('Right ascension of Epimetheus (deg, actual)',
                                                    test * constants.DPR)

    test = bp.center_right_ascension('epimetheus', apparent=True)
    show_info('Right ascension of Epimetheus (deg, apparent)',
                                                    test * constants.DPR)

    ########################

    if printing: print('\n********* declination')

    test = bp.declination(apparent=False)
    show_info('Declination (deg, actual)', test * constants.DPR)

    test = bp.declination(apparent=True)
    show_info('Declination (deg, apparent)', test * constants.DPR)

    test = bp.center_declination('saturn', apparent=False)
    show_info('Declination of Saturn (deg, actual)', test * constants.DPR)

    test = bp.center_declination('saturn', apparent=True)
    show_info('Declination of Saturn (deg, apparent)', test * constants.DPR)

    test = bp.center_declination('epimetheus', apparent=False)
    show_info('Declination of Epimetheus (deg, actual)',
                                                    test * constants.DPR)

    test = bp.center_declination('epimetheus', apparent=True)
    show_info('Declination of Epimetheus (deg, apparent)',
                                                    test * constants.DPR)

    ########################

    if printing: print('\n********* celestial and polar angles')

    test = bp.celestial_north_angle()
    show_info('Celestial north angle (deg)', test * constants.DPR)

    test = bp.celestial_east_angle()
    show_info('Celestial east angle (deg)', test * constants.DPR)

    test = bp.pole_clock_angle('saturn')
    show_info('Saturn pole clock angle (deg)', test * constants.DPR)

    test = bp.pole_position_angle('saturn')
    show_info('Saturn pole position angle (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* observer distances')

    test = bp.distance('saturn')
    show_info('Distance observer to Saturn (km)', test)

    test = bp.distance('saturn', direction='dep')
    show_info('Distance observer to Saturn via dep (km)', test)

    test = bp.center_distance('saturn')
    show_info('Distance observer to Saturn center (km)', test)

    test = bp.distance('saturn_main_rings')
    show_info('Distance observer to rings (km)', test)

    test = bp.center_distance('saturn_main_rings')
    show_info('Distance observer to ring center (km)', test)

    test = bp.distance('saturn:limb')
    show_info('Distance observer to Saturn limb (km)', test)

    test = bp.distance('saturn:ansa')
    show_info('Distance observer to ansa (km)', test)

    test = bp.distance('epimetheus')
    show_info('Distance observer to Epimetheus (km)', test)

    test = bp.center_distance('epimetheus')
    show_info('Distance observer to Epimetheus center (km)', test)

    ########################

    if printing: print('\n********* Sun distances')

    test = bp.distance('saturn', direction='arr')
    show_info('Distance Sun to Saturn, arrival (km)', test)

    test = bp.distance(('sun', 'saturn'), direction='dep')
    show_info('Distance Sun to Saturn, departure (km)', test)

    test = bp.center_distance('saturn', direction='arr')
    show_info('Distance Sun to Saturn center, arrival (km)', test)

    test = bp.center_distance(('sun', 'saturn'), direction='dep')
    show_info('Distance Sun to Saturn center, departure (km)', test)

    test = bp.distance('saturn_main_rings', direction='arr')
    show_info('Distance Sun to rings, arrival (km)', test)

    test = bp.distance(('sun', 'saturn_main_rings'), direction='dep')
    show_info('Distance Sun to rings, departure (km)', test)

    test = bp.center_distance('saturn_main_rings', direction='arr')
    show_info('Distance Sun to ring center, arrival (km)', test)

    test = bp.center_distance(('sun', 'saturn_main_rings'), direction='dep')
    show_info('Distance Sun to ring center, departure (km)', test)

    test = bp.distance('saturn:ansa', direction='arr')
    show_info('Distance Sun to ansa (km)', test)

    test = bp.distance('saturn:limb', direction='arr')
    show_info('Distance Sun to limb (km)', test)

    ########################

    if printing: print('\n********* observer light time')

    test = bp.light_time('saturn')
    show_info('Light-time observer to Saturn (sec)', test)

    test = bp.light_time('saturn', direction='dep')
    show_info('Light-time observer to Saturn via dep (sec)', test)

    test = bp.light_time('saturn_main_rings')
    show_info('Light-time observer to rings (sec)', test)

    test = bp.light_time('saturn:limb')
    show_info('Light-time observer to limb (sec)', test)

    test = bp.light_time('saturn:ansa')
    show_info('Light-time observer to ansa (sec)', test)

    test = bp.center_light_time('saturn')
    show_info('Light-time observer to Saturn center (sec)', test)

    test = bp.center_light_time('saturn_main_rings')
    show_info('Light-time observer to ring center (sec)', test)

    test = bp.light_time('epimetheus')
    show_info('Light-time observer to Epimetheus (sec)', test)

    test = bp.center_light_time('epimetheus')
    show_info('Light-time observer to Epimetheus center (sec)', test)

    ########################

    if printing: print('\n********* Sun light time')

    test = bp.light_time('saturn', direction='arr')
    show_info('Light-time Sun to Saturn via arr (sec)', test)

    test = bp.light_time(('sun', 'saturn'))
    show_info('Light-time Sun to Saturn via sun-saturn (sec)', test)

    test = bp.center_light_time(('sun', 'saturn'))
    show_info('Light-time Sun to Saturn at centers (sec)', test)

    test = bp.light_time('saturn_main_rings', direction='arr')
    show_info('Light-time Sun to rings (sec)', test)

    test = bp.center_light_time(('sun', 'saturn_main_rings'))
    show_info('Light-time Sun to rings at centers (sec)', test)

    test = bp.light_time('saturn:ansa', direction='arr')
    show_info('Light-time Sun to ansa (sec)', test)

    test = bp.light_time('saturn:limb', direction='arr')
    show_info('Light-time Sun to limb (sec)', test)

    ########################

    if printing: print('\n********* event time')

    test = bp.event_time(())
    show_info('Event time at Cassini (sec, TDB)', test)

    test = bp.event_time('saturn')
    show_info('Event time at Saturn (sec, TDB)', test)

    test = bp.event_time('saturn_main_rings')
    show_info('Event time at rings (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus (sec, TDB)', test)

    test = bp.center_time(())
    show_info('Event time at Cassini center (sec, TDB)', test)

    test = bp.center_time('saturn')
    show_info('Event time at Saturn (sec, TDB)', test)

    test = bp.center_time('saturn_main_rings')
    show_info(' Event time at ring center (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus center (sec, TDB)', test)

    ########################

    if printing: print('\n********* resolution')

    test = bp.resolution('saturn', 'u')
    show_info('Saturn resolution along u axis (km)', test)

    test = bp.resolution('saturn', 'v')
    show_info('Saturn resolution along v axis (km)', test)

    test = bp.center_resolution('saturn', 'u')
    show_info('Saturn center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn', 'v')
    show_info('Saturn center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn')
    show_info('Saturn finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn')
    show_info('Saturn coarsest resolution (km)', test)

    test = bp.resolution('epimetheus', 'u')
    show_info('Epimetheus resolution along u axis (km)', test)

    test = bp.resolution('epimetheus', 'v')
    show_info('Epimetheus resolution along v axis (km)', test)

    test = bp.center_resolution('epimetheus', 'u')
    show_info('Epimetheus center resolution along u axis (km)', test)

    test = bp.center_resolution('epimetheus', 'v')
    show_info('Epimetheus center resolution along v axis (km)', test)

    test = bp.finest_resolution('epimetheus')
    show_info('Epimetheus finest resolution (km)', test)

    test = bp.coarsest_resolution('epimetheus')
    show_info('Epimetheus coarsest resolution (km)', test)

    test = bp.resolution('saturn_main_rings', 'u')
    show_info('Ring resolution along u axis (km)', test)

    test = bp.resolution('saturn_main_rings', 'v')
    show_info('Ring resolution along v axis (km)', test)

    test = bp.center_resolution('saturn_main_rings', 'u')
    show_info('Ring center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn_main_rings', 'v')
    show_info('Ring center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn_main_rings')
    show_info('Ring finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn_main_rings')
    show_info('Ring coarsest resolution (km)', test)

    test = bp.ring_radial_resolution('saturn_main_rings')
    show_info('Ring radial resolution (km)', test)

    test = bp.ring_angular_resolution('saturn_main_rings')
    show_info('Ring angular resolution (deg)', test * constants.DPR)

    radii = bp.ring_radius('saturn_main_rings')
    show_info('Ring angular resolution (km)', test * radii)

    test = bp.resolution('saturn:ansa', 'u')
    show_info('Ansa resolution along u axis (km)', test)

    test = bp.resolution('saturn:ansa', 'v')
    show_info('Ansa resolution along v axis (km)', test)

    test = bp.center_resolution('saturn:ansa', 'u')
    show_info('Ansa center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn:ansa', 'v')
    show_info('Ansa center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn:ansa')
    show_info('Ansa finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn:ansa')
    show_info('Ansa coarsest resolution (km)', test)

    test = bp.ansa_radial_resolution('saturn:ansa')
    show_info('Ansa radial resolution (km)', test)

    test = bp.ansa_vertical_resolution('saturn:ansa')
    show_info('Ansa vertical resolution (km)', test)

    test = bp.resolution('saturn:limb', 'u')
    show_info('Limb resolution along u axis (km)', test)

    test = bp.resolution('saturn:limb', 'v')
    show_info('Limb resolution along v axis (km)', test)

    test = bp.resolution('saturn:limb', 'u')
    show_info('Limb resolution along u axis (km)', test)

    test = bp.resolution('saturn:limb', 'v')
    show_info('Limb resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn:limb')
    show_info('Limb finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn:limb')
    show_info('Limb coarsest resolution (km)', test)

    ########################

    if printing: print('\n********* surface latitude')

    test = bp.latitude('saturn', lat_type='centric')
    show_info('Saturn latitude, planetocentric (deg)', test * constants.DPR)

    test = bp.latitude('saturn', lat_type='squashed')
    show_info('Saturn latitude, squashed (deg)', test * constants.DPR)

    test = bp.latitude('saturn', lat_type='graphic')
    show_info('Saturn latitude, planetographic (deg)', test * constants.DPR)

    test = bp.sub_observer_latitude('saturn')
    show_info('Saturn sub-observer latitude (deg)', test * constants.DPR)

    test = bp.sub_solar_latitude('saturn')
    show_info('Saturn sub-solar latitude (deg)', test * constants.DPR)

    test = bp.latitude('epimetheus', lat_type='centric')
    show_info('Epimetheus latitude, planetocentric (deg)', test * constants.DPR)

    test = bp.latitude('epimetheus', lat_type='squashed')
    show_info('Epimetheus latitude, squashed (deg)', test * constants.DPR)

    test = bp.latitude('epimetheus', lat_type='graphic')
    show_info('Epimetheus latitude, planetographic (deg)', test * constants.DPR)

    test = bp.sub_observer_latitude('epimetheus')
    show_info('Epimetheus sub-observer latitude (deg)', test * constants.DPR)

    test = bp.sub_solar_latitude('epimetheus')
    show_info('Epimetheus sub-solar latitude (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* surface longitude')

    test = bp.longitude('saturn')
    show_info('Saturn longitude (deg)', test * constants.DPR)

    test = bp.longitude('saturn', reference='iau')
    show_info('Saturn longitude wrt IAU frame (deg)',
                                                test * constants.DPR)

    test = bp.longitude('saturn', lon_type='centric')
    show_info('Saturn longitude centric (deg)', test * constants.DPR)

    test = bp.longitude('saturn', lon_type='graphic')
    show_info('Saturn longitude graphic (deg)', test * constants.DPR)

    test = bp.longitude('saturn', lon_type='squashed')
    show_info('Saturn longitude squashed (deg)', test * constants.DPR)

    test = bp.longitude('saturn', direction='east')
    show_info('Saturn longitude eastward (deg)', test * constants.DPR)

    test = bp.longitude('saturn', minimum=-180)
    show_info('Saturn longitude with -180 minimum (deg)',
                                                test * constants.DPR)

    test = bp.longitude('saturn', reference='iau', minimum=-180)
    show_info('Saturn longitude wrt IAU frame with -180 minimum (deg)',
                                                test * constants.DPR)

    test = bp.longitude('saturn', reference='sun')
    show_info('Saturn longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.longitude('saturn', reference='sha')
    show_info('Saturn longitude wrt SHA (deg)', test * constants.DPR)

    test = bp.longitude('saturn', reference='obs')
    show_info('Saturn longitude wrt observer (deg)',
                                                test * constants.DPR)

    test = bp.longitude('saturn', reference='oha')
    show_info('Saturn longitude wrt OHA (deg)', test * constants.DPR)

    test = bp.sub_observer_longitude('saturn', reference='iau')
    show_info('Saturn sub-observer longitude wrt IAU (deg)',
                                                test * constants.DPR)

    test = bp.sub_observer_longitude('saturn', reference='sun', minimum=-180)
    show_info('Saturn sub-observer longitude wrt Sun (deg)',
                                                test * constants.DPR)

    test = bp.sub_observer_longitude('saturn', reference='obs', minimum=-180)
    show_info('Saturn sub-observer longitude wrt observer (deg)',
                                                test * constants.DPR)

    test = bp.sub_solar_longitude('saturn', reference='iau')
    show_info('Saturn sub-solar longitude wrt IAU (deg)',
                                                test * constants.DPR)

    test = bp.sub_solar_longitude('saturn', reference='obs', minimum=-180)
    show_info('Saturn sub-solar longitude wrt observer (deg)',
                                                test * constants.DPR)

    test = bp.sub_solar_longitude('saturn', reference='sun', minimum=-180)
    show_info('Saturn sub-solar longitude wrt Sun (deg)',
                                                test * constants.DPR)

    test = bp.longitude('epimetheus')
    show_info('Epimetheus longitude (deg)', test * constants.DPR)

    test = bp.sub_observer_longitude('epimetheus')
    show_info('Epimetheus sub-observer longitude (deg)', test * constants.DPR)

    test = bp.sub_solar_longitude('epimetheus')
    show_info('Epimetheus sub-solar longitude (deg)', test * constants.DPR)

# Used for testing other images
#     test = bp.longitude('enceladus')
#     show_info('Enceladus longitude (deg)', test * constants.DPR)
#
#     test = bp.sub_observer_longitude('enceladus')
#     show_info('Enceladus sub-observer longitude (deg)', test * constants.DPR)
#
#     test = bp.sub_solar_longitude('enceladus')
#     show_info('Enceladus sub-solar longitude (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* surface incidence, emission, phase')

    test = bp.phase_angle('saturn')
    show_info('Saturn phase angle (deg)', test * constants.DPR)

    test = bp.scattering_angle('saturn')
    show_info('Saturn scattering angle (deg)', test * constants.DPR)

    test = bp.incidence_angle('saturn')
    show_info('Saturn incidence angle (deg)', test * constants.DPR)

    test = bp.emission_angle('saturn')
    show_info('Saturn emission angle (deg)', test * constants.DPR)

    test = bp.lambert_law('saturn')
    show_info('Saturn as a Lambert law', test)

    ########################

    if printing: print('\n********* ring radius, radial modes')

    test = bp.ring_radius('saturn_main_rings')
    show_info('Ring radius (km)', test)

    test0 = bp.ring_radius('saturn_main_rings', 70.e3, 100.e3)
    show_info('Ring radius, 70-100 kkm (km)', test0)

    test1 = bp.radial_mode(test0.key, 40, 0., 1000., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, mode 1 (km)', test1)

    test = bp.radial_mode(test1.key, 40, 0., -1000., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, mode 1 canceled (km)', test)

    test2 = bp.radial_mode(test1.key, 25, 0., 500., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, modes 1 and 2 (km)', test2)

    test = bp.ring_radius('saturn_main_rings').without_mask()
    show_info('Ring radius unmasked (km)', test)

    ########################

    if printing: print('\n********* ring longitude, azimuth')

    test = bp.ring_longitude('saturn_main_rings', reference='node')
    show_info('Ring longitude wrt node (deg)', test * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', 'node', 70.e3, 100.e3)
    show_info('Ring longitude wrt node, 70-100 kkm (deg)', test * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='aries')
    show_info('Ring longitude wrt Aries (deg)', test * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='obs')
    show_info('Ring longitude wrt observer (deg)',
                                                test * constants.DPR)

    test = bp.ring_azimuth('saturn_main_rings', 'obs')
    show_info('Ring azimuth wrt observer (deg)', test * constants.DPR)

    test = bp.ring_azimuth('saturn:ring', 'obs')
    show_info('Ring azimuth wrt observer (deg)', test * constants.DPR)

    compare = bp.ring_longitude('saturn_main_rings', 'obs')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt observer (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_azimuth('saturn:ring', 'obs')
    show_info('Ring azimuth wrt observer, unmasked (deg)', test * constants.DPR)

    compare = bp.ring_longitude('saturn:ring', 'obs')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt observer, unmasked (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='oha')
    show_info('Ring longitude wrt OHA (deg)', test * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='sun')
    show_info('Ring longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.ring_azimuth('saturn_main_rings', reference='sun')
    show_info('Ring azimuth wrt Sun (deg)', test * constants.DPR)

    compare = bp.ring_longitude('saturn_main_rings', 'sun')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt Sun (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_azimuth('saturn:ring', reference='sun')
    show_info('Ring azimuth wrt Sun, unmasked (deg)', test * constants.DPR)

    compare = bp.ring_longitude('saturn:ring', 'sun')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt Sun, unmasked (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='sha')
    show_info('Ring longitude wrt SHA (deg)', test * constants.DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'node')
    show_info('Ring sub-observer longitude wrt node (deg)',
                                                        test * constants.DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'aries')
    show_info('Ring sub-observer longitude wrt Aries (deg)',
                                                        test * constants.DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'sun')
    show_info('Ring sub-observer longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'obs')
    show_info('Ring sub-observer longitude wrt observer (deg)',
                                                        test * constants.DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'node')
    show_info('Ring sub-solar longitude wrt node (deg)', test * constants.DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'aries')
    show_info('Ring sub-solar longitude wrt Aries (deg)', test * constants.DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'sun')
    show_info('Ring sub-solar longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'obs')
    show_info('Ring sub-solar longitude wrt observer (deg)',
                                                        test * constants.DPR)

    ########################

    if printing: print('\n********* ring phase angle')

    test = bp.phase_angle('saturn_main_rings')
    show_info('Ring phase angle (deg)', test * constants.DPR)

    test = bp.sub_observer_longitude('saturn_main_rings', 'sun', minimum=-180)
    show_info('Ring observer-sun longitude (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* ring incidence, solar elevation')

    test = bp.ring_incidence_angle('saturn_main_rings', 'sunward')
    show_info('Ring incidence angle, sunward (deg)', test * constants.DPR)

    test = bp.ring_incidence_angle('saturn_main_rings', 'north')
    show_info('Ring incidence angle, north (deg)', test * constants.DPR)

    test = bp.ring_incidence_angle('saturn_main_rings', 'prograde')
    show_info('Ring incidence angle, prograde (deg)', test * constants.DPR)

    test = bp.incidence_angle('saturn_main_rings')
    show_info('Ring incidence angle via incidence() (deg)',
                                                        test * constants.DPR)

    test = bp.ring_elevation('saturn_main_rings', reference='sun')
    show_info('Ring elevation wrt Sun (deg)', test * constants.DPR)

    compare = bp.ring_incidence_angle('saturn_main_rings', 'north')
    diff = test + compare
    show_info('Ring elevation wrt Sun plus north incidence (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'sunward')
    show_info('Ring center incidence angle, sunward (deg)',
                                                        test * constants.DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'north')
    show_info('Ring center incidence angle, north (deg)', test * constants.DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'prograde')
    show_info('Ring center incidence angle, prograde (deg)',
                                                        test * constants.DPR)

    test = bp.ring_elevation('saturn:ring', reference='sun')
    show_info('Ring elevation wrt Sun, unmasked (deg)', test * constants.DPR)

    compare = bp.ring_incidence_angle('saturn:ring', 'north')
    diff = test + compare
    show_info('Ring elevation wrt Sun plus north incidence, unmasked (deg)',
                                                        diff * constants.DPR)

    ########################

    if printing: print('\n********* ring emission, observer elevation')

    test = bp.ring_emission_angle('saturn_main_rings', 'sunward')
    show_info('Ring emission angle, sunward (deg)', test * constants.DPR)

    test = bp.ring_emission_angle('saturn_main_rings', 'north')
    show_info('Ring emission angle, north (deg)', test * constants.DPR)

    test = bp.ring_emission_angle('saturn_main_rings', 'prograde')
    show_info('Ring emission angle, prograde (deg)', test * constants.DPR)

    test = bp.emission_angle('saturn_main_rings')
    show_info('Ring emission angle via emission() (deg)', test * constants.DPR)

    test = bp.ring_elevation('saturn_main_rings', reference='obs')
    show_info('Ring elevation wrt observer (deg)', test * constants.DPR)

    compare = bp.ring_emission_angle('saturn_main_rings', 'north')
    diff = test + compare
    show_info('Ring elevation wrt observer plus north emission (deg)',
                                                        diff * constants.DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'sunward')
    show_info('Ring center emission angle, sunward (deg)', test * constants.DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'north')
    show_info('Ring center emission angle, north (deg)', test * constants.DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'prograde')
    show_info('Ring center emission angle, prograde (deg)',
                                                        test * constants.DPR)

    test = bp.ring_elevation('saturn:ring', reference='obs')
    show_info('Ring elevation wrt observer, unmasked (deg)',
                                                        test * constants.DPR)

    compare = bp.ring_emission_angle('saturn:ring', 'north')
    diff = test + compare
    show_info('Ring elevation wrt observer plus north emission, unmasked (deg)',
                                                        diff * constants.DPR)

    ########################

    if printing: print('\n********* ansa geometry')

    test = bp.ansa_radius('saturn:ansa')
    show_info('Ansa radius (km)', test)

    test = bp.ansa_altitude('saturn:ansa')
    show_info('Ansa altitude (km)', test)

    test = bp.ansa_longitude('saturn:ansa', 'node')
    show_info('Ansa longitude wrt node (deg)', test * constants.DPR)

    test = bp.ansa_longitude('saturn:ansa', 'aries')
    show_info('Ansa longitude wrt Aries (deg)', test * constants.DPR)

    test = bp.ansa_longitude('saturn:ansa', 'obs')
    show_info('Ansa longitude wrt observer (deg)', test * constants.DPR)

    test = bp.ansa_longitude('saturn:ansa', 'oha')
    show_info('Ansa longitude wrt OHA (deg)', test * constants.DPR)

    test = bp.ansa_longitude('saturn:ansa', 'sun')
    show_info('Ansa longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.ansa_longitude('saturn:ansa', 'sha')
    show_info('Ansa longitude wrt SHA (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* limb altitude')

    test = bp.limb_altitude('saturn:limb')
    show_info('Limb altitude (km)', test)

    ########################

    if printing: print('\n********* limb longitude')

    test = bp.longitude('saturn:limb', 'iau')
    show_info('Limb longitude wrt IAU (deg)', test * constants.DPR)

    test = bp.longitude('saturn:limb', 'obs')
    show_info('Limb longitude wrt observer (deg)', test * constants.DPR)

    test = bp.longitude('saturn:limb', reference='obs', minimum=-180)
    show_info('Limb longitude wrt observer, -180 (deg)',
                                                    test * constants.DPR)

    test = bp.longitude('saturn:limb', 'oha')
    show_info('Limb longitude wrt OHA (deg)', test * constants.DPR)

    test = bp.longitude('saturn:limb', 'sun')
    show_info('Limb longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.longitude('saturn:limb', 'sha')
    show_info('Limb longitude wrt SHA (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* limb latitude')

    test = bp.latitude('saturn:limb', lat_type='centric')
    show_info('Limb planetocentric latitude (deg)', test * constants.DPR)

    test = bp.latitude('saturn:limb', lat_type='squashed')
    show_info('Limb squashed latitude (deg)', test * constants.DPR)

    test = bp.latitude('saturn:limb', lat_type='graphic')
    show_info('Limb planetographic latitude (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* orbit longitude')

    test = bp.orbit_longitude('epimetheus', reference='obs')
    show_info('Epimetheus orbit longitude wrt observer (deg)', test * constants.DPR)

    test = bp.orbit_longitude('epimetheus', reference='oha')
    show_info('Epimetheus orbit longitude wrt OHA (deg)', test * constants.DPR)

    test = bp.orbit_longitude('epimetheus', reference='sun')
    show_info('Epimetheus orbit longitude wrt Sun (deg)', test * constants.DPR)

    test = bp.orbit_longitude('epimetheus', reference='sha')
    show_info('Epimetheus orbit longitude wrt SHA (deg)', test * constants.DPR)

    test = bp.orbit_longitude('epimetheus', reference='aries')
    show_info('Epimetheus orbit longitude wrt Aries (deg)', test * constants.DPR)

    test = bp.orbit_longitude('epimetheus', reference='node')
    show_info('Epimetheus orbit longitude wrt node (deg)', test * constants.DPR)

    ########################

    if printing: print('\n********* masks')

    test = bp.where_intercepted('saturn')
    show_info('Mask of Saturn intercepted', test)

    test = bp.evaluate(('where_intercepted', 'saturn'))
    show_info('Mask of Saturn intercepted via evaluate()', test)

    test = bp.where_sunward('saturn')
    show_info('Mask of Saturn sunward', test)

    test = bp.evaluate(('where_sunward', 'saturn'))
    show_info('Mask of Saturn sunward via evaluate()', test)

    test = bp.where_below(('incidence_angle', 'saturn'), constants.HALFPI)
    show_info('Mask of Saturn sunward via where_below()', test)

    test = bp.where_antisunward('saturn')
    show_info('Mask of Saturn anti-sunward', test)

    test = bp.where_above(('incidence_angle', 'saturn'), constants.HALFPI)
    show_info('Mask of Saturn anti-sunward via where_above()', test)

    test = bp.where_between(('incidence_angle', 'saturn'), constants.HALFPI,3.2)
    show_info('Mask of Saturn anti-sunward via where_between()', test)

    test = bp.where_in_front('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn in front of rings', test)

    test = bp.where_in_back('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn behind rings', test)

    test = bp.where_inside_shadow('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn in shadow of rings', test)

    test = bp.where_outside_shadow('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn outside shadow of rings', test)

    test = bp.where_intercepted('saturn_main_rings')
    show_info('Mask of rings intercepted', test)

    test = bp.where_in_front('saturn_main_rings', 'saturn')
    show_info('Mask of rings in front of Saturn', test)

    test = bp.where_in_back('saturn_main_rings', 'saturn')
    show_info('Mask of rings behind Saturn', test)

    test = bp.where_inside_shadow('saturn_main_rings', 'saturn')
    show_info('Mask of rings in shadow of Saturn', test)

    test = bp.where_outside_shadow('saturn_main_rings', 'saturn')
    show_info('Mask of rings outside shadow of Saturn', test)

    test = bp.where_sunward('saturn_main_rings')
    show_info('Mask of rings sunward', test)

    test = bp.where_antisunward('saturn_main_rings')
    show_info('Mask of rings anti-sunward', test)

    ########################

    if printing: print('\n********* borders')

    mask = bp.where_intercepted('saturn')
    test = bp.border_inside(mask)
    show_info('Border of Saturn intercepted mask, inside', test)

    test = bp.border_outside(mask)
    show_info('Border of Saturn intercepted mask, outside', test)

    test = bp.border_below(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius below 100 kkm', test)

    test = bp.border_atop(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius atop 100 kkm', test)

    test = bp.border_above(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius above 100 kkm', test)

    test = bp.evaluate(('border_above', ('ring_radius', 'saturn:ring'), 100.e3))
    show_info('Border of ring radius above 100 kkm via evaluate()', test)

    ########################

    if printing: print('\n********* EMPTY EVENTS')

    test = bp.where_below(('ring_radius', 'saturn_main_rings'), 10.e3)
    show_info('Empty mask of Saturn ring radius below 10 kkm', test)

    test = bp.ring_radius('pluto:ring')
    show_info('Empty ring radius for Pluto (km)', test)

    test = bp.longitude('pluto')
    show_info('Empty longitude for Pluto (km)', test)

    test = bp.incidence_angle('pluto')
    show_info('Empty incidence angle for Pluto (km)', test)

    config.LOGGING.off()

    return bp

################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.config import ABERRATION

UNITTEST_SATURN_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/W1573721822_1.IMG')
UNITTEST_RHEA_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/N1649465464_1.IMG')
UNITTEST_UNDERSAMPLE = 16

class Test_Backplane(unittest.TestCase):

    OLD_RHEA_SURFACE = None

    def setUp(self):
        global OLD_RHEA_SURFACE

        import oops.body
        from oops.surface_.ellipsoid  import Ellipsoid

        oops.body.Body.reset_registry()
        oops.body.define_solar_system('2000-01-01', '2020-01-01')

        # Distort Rhea's shape for better Ellipsoid testing
        rhea = Body.as_body('RHEA')
        OLD_RHEA_SURFACE = rhea.surface
        old_rhea_radii = OLD_RHEA_SURFACE.radii

        new_rhea_radii = tuple(np.array([1.1, 1., 0.9]) * old_rhea_radii)
        new_rhea_surface = Ellipsoid(rhea.path, rhea.frame, new_rhea_radii)
        Body.as_body('RHEA').surface = new_rhea_surface

  #      config.LOGGING.on('   ')
        config.EVENT_CONFIG.collapse_threshold = 0.
        config.SURFACE_PHOTONS.collapse_threshold = 0.

    def tearDown(self):
        global OLD_RHEA_SURFACE

        config.LOGGING.off()
        config.EVENT_CONFIG.collapse_threshold = 3.
        config.SURFACE_PHOTONS.collapse_threshold = 3.

        # Restore Rhea's shape
        Body.as_body('RHEA').surface = OLD_RHEA_SURFACE

        ABERRATION.old = False

    def runTest(self):
      import oops.inst.cassini.iss as iss

      from oops.surface_.spheroid  import Spheroid
      from oops.surface_.ellipsoid  import Ellipsoid
      from oops.surface_.centricspheroid import CentricSpheroid
      from oops.surface_.graphicspheroid import GraphicSpheroid
      from oops.surface_.centricellipsoid import CentricEllipsoid
      from oops.surface_.graphicellipsoid import GraphicEllipsoid

      for ABERRATION.old in (False, True):

        snap = iss.from_file(UNITTEST_SATURN_FILESPEC, fast_distortion=False)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        #meshgrid = Meshgrid(snap.fov, (512,512))
        uv0 = meshgrid.uv
        bp = Backplane(snap, meshgrid)

        #### Actual (ra,dec)
        ra = bp.right_ascension(apparent=False)
        dec = bp.declination(apparent=False)

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        ev.neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec)
        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-9)

        #### Apparent (ra,dec)  # test doesn't work for ABERRATION=old
        if not ABERRATION.old:
            ra = bp.right_ascension(apparent=True)
            dec = bp.declination(apparent=True)

            ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
            ev.neg_arr_ap_j2000 = Vector3.from_ra_dec_length(ra, dec)
            uv = snap.fov.uv_from_los(ev.neg_arr_ap)

            diff = uv - uv0
            self.assertTrue(diff.norm().max() < 1.e-9)

        # RingPlane (rad, lon)
        rad = bp.ring_radius('saturn:ring')
        lon = bp.ring_longitude('saturn:ring', reference='node')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN_RING_PLANE')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (rad,lon))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # Ansa (rad, alt)
        rad = bp.ansa_radius('saturn:ansa', radius_type='right')
        alt = bp.ansa_altitude('saturn:ansa')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN_RING_PLANE')
        surface = Ansa.for_ringplane(body.surface)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (rad,alt))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # Spheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='squashed')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # CentricSpheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='centric')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        surface = CentricSpheroid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # GraphicSpheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='graphic')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        surface = GraphicSpheroid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        ######## Rhea tests, with Rhea modified
        body = Body.as_body('RHEA')
        snap = iss.from_file(UNITTEST_RHEA_FILESPEC, fast_distortion=False)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)

        uv0 = meshgrid.uv
        bp = Backplane(snap, meshgrid)

        # Ellipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='squashed')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='squashed')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

        # CentricEllipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='centric')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        surface = CentricEllipsoid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

        # GraphicEllipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='graphic')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='graphic')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        surface = GraphicEllipsoid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

########################################

class Test_Backplane_Exercises(unittest.TestCase):

    def runTest(self):

        import oops.inst.cassini.iss as iss
        iss.initialize(asof='2019-09-01', mst_pck=True)

        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                'cassini/ISS/W1573721822_1.IMG')

#         filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                 'cassini/ISS/W1575632515_1.IMG')
# TARGET = SATURN

#         filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                 'cassini/ISS/N1573845439_1.IMG')
# TARGET = 'ENCELADUS'

        TEST_LEVEL = 3

        logging = False         # Turn on for efficiency testing

        if TEST_LEVEL == 3:     # long and slow, creates images, logging off
            printing = True
            saving = True
            undersample = 1
        elif TEST_LEVEL == 2:   # faster, prints info, no images, undersampled
            printing = True
            saving = False
            undersample = 16
        elif TEST_LEVEL == 1:   # executes every routine but does no printing
            printing = False
            saving = False
            undersample = 32

        if TEST_LEVEL > 0:

            bp = exercise_backplanes(filespec, printing, logging, saving,
                                     undersample,
                                     use_inventory=True, inventory_border=4)

        else:
            print('test skipped')

########################################

if __name__ == '__main__':

    unittest.main(verbosity=2)

################################################################################
