################################################################################
# oops/backplane/__init__.py: Backplane class
################################################################################

import datetime
import functools
import numpy as np
import types

from polymath               import Boolean, Qube, Scalar, Vector3
from oops.body              import Body
from oops.config            import LOGGING
from oops.surface.ansa      import Ansa
from oops.surface.limb      import Limb
from oops.surface.ringplane import RingPlane

class Backplane(object):
    """Class that supports the generation and manipulation of sets of backplanes
    with a particular observation.

    intermediate results are cached to speed up calculations.
    """

    DIAGNOSTICS = False     # set True to log diagnostics
    PERFORMANCE = False     # set True to log timings of surface calculations

    ALL_DERIVS = False

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

        Notes about Backplane Arrays

        Every backplane array method takes an "event_key" as its first input.
        This is normally indicated by a tuple of two items:
            (source_key, surface_key)
        where the first item is the source of the lighting (usually the Sun) and
        the second is the surface for which geometry is needed.

        Each intercepted surface is defined by a surface key string, one of:
            body_name           for the default surface of a body;
            body_name:RING      for the ring surface associated with a body;
            body_name:ANSA      for the ansa surface associated with a body;
            body_name:LIMB      for the limb surface associated with a body.

        The light source is defined by a source key string of the form:
            body_name<          for dispersed illumination;
            body_name>          for occultation illumination;
            body_name-          for path-based illumination.

        These strings are not case-sensitive.

        As a shortcut, you can specify a surface_key string alone in place of an
        event_key, in which case "SUN<" is assumed as the source.

        In dispersed illumination, each backplane has spatial dimensions that
        are defined by the Meshgrid. Photons leave the source in all directions,
        and some of those that intercept the surface are reflected toward the
        detector. The detector selects the photons it receives based on its. The
        event is defined as the moment the photons hit the surface and are
        reflected.

        In occultation illumination, backplanes have no spatial dimensions.
        Photons leave the source along a direct line of sight to the detector,
        and the event is defined as the time and location where the photons
        intercept the surface.

        In path-based illumination, the photon follows one or more straight-
        line paths between the origin points of the surfaces. The backplanes
        have no spatial dimensions, but can be used to answer such questions as
        "what is the sub-solar latitude on Saturn?" or "where is Enceladus in
        this image?"

        Examples:
        ('SUN<', 'SATURN:RING') could describe a 2-D image of Saturn's rings.
        ('SUN>', 'SATURN:RING') could describe a solar occultation profile of
                Saturn's rings.
        ('SUN-', 'SATURN:RING') defines the direction of the center of Saturn's
                rings (which is also the center of Saturn).

        Most of the time, an event key only contains two items as in the
        examples above. However, shadowing can be defined by inserting one
        additional surface key after 'SUN<'. For example,
            ('SUN<', 'MIMAS', 'SATURN:RING')
        describes the surface event at Mimas, subject to the constraint that the
        photons subsequently reflected off of Saturn and arrived at the
        detector; it can be used to determine how Mimas shadows the rings.
        """

        self.obs = obs

        # Establish spatial and temporal grids
        self._input_meshgrid = meshgrid
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
        self._input_inventory = inventory
        if isinstance(inventory, dict):
            self.inventory = inventory
        elif inventory:
            self.inventory = {}
        else:
            self.inventory = None

        self.inventory_border = inventory_border

        # Define events
        self.obs_event = obs.event_at_grid(self.meshgrid, time=self.time)

        # dict[derivs] = event
        self.obs_events = {
            False: self.obs_event.wod,
            True : self.obs_event.with_los_derivs()
        }

        self.obs_gridless_event = obs.gridless_event(self.meshgrid,
                                                     time=self.time)

        self.shape = self.obs_event.shape

        # The surface_events dictionary comes in two versions, with and without
        # derivatives with respect to los and time.

        self.surface_events = {
            False: {},
            True : {}
        }

        # Gridless/occultation events of photon paths arriving at the detector
        self.gridless_arrivals = {}

        # The backplanes dictionary holds every backplane that has been
        # calculated. This includes boolean backplanes, aka masks. A backplane
        # is keyed by (name of backplane, event_key, optional additional
        # parameters). The name of the backplane is always the name of the
        # backplane method that generates this backplane. For example,
        # ('phase_angle', ('SATURN',)) is the key for a backplane of phase angle
        # values at the Saturn intercept points.
        #
        # If the function that returns a backplane requires additional
        # parameters, those appear in the tuple after the event key in the same
        # order that they appear in the calling function. For example,
        # ('latitude', ('SATURN',), 'graphic') is the key for the backplane of
        # planetographic latitudes at Saturn.

        self.backplanes = {}
        self.backplanes_with_derivs = {}    # used by ALL_DERIVS option

        # Antimasks of surfaces, keyed by surface key.
        self.antimasks = {}

        # We save unmasked surface intercept events based on the intercept_key
        # of the surface. This avoids the re-calculating of intercept events
        # when the only change is to their coordinates or mask. The dictionary
        # key is an event_key in which each surface_key (after the first item,
        # which is the light source name) is replaced by the unmasked surface
        # intercept key.

        self.intercepts = {
            False: {},
            True : {},
        }

    ############################################################################
    # Serialization support
    # For quicker backplane generation after unpickling, we save the various
    # internal buffers, but try to avoid any duplication
    ############################################################################

    def __getstate__(self):

        return (self.obs, self._input_meshgrid, self.time,
                self._input_inventory, self.inventory_border,
                self.surface_events, self.gridless_arrivals, self.antimasks,
                self.intercepts)

    def __setstate__(self, state):

        (obs, meshgrid, time, inventory, inventory_border,
         surface_events, gridless_arrivals, antimasks, intercepts) = state

        self.__init__(obs, meshgrid, time, inventory, inventory_border)
        self.surface_events = surface_events
        self.gridless_arrivals = gridless_arrivals
        self.antimasks = antimasks
        self.intercepts = intercepts

    ############################################################################
    # Resolution properties
    ############################################################################

    @property
    def dlos_duv(self):
        if not hasattr(self, '_dlos_duv'):
            self._dlos_duv = self.meshgrid.dlos_duv(self.time)

        return self._dlos_duv

    @property
    def dlos_duv1(self):
        """The derivative of the line of sight with respect to (u1,v1), where
        (u1,v1) match (u,v) but have been forced to be orthogonal. This is done
        by leaving the lesser pixel size alone, and shifting the greater pixel
        edge to be orthogonal while conserving the pixel area. This is a
        better pair to use for determining spatial resolution.
        """

        if hasattr(self, '_dlos_duv1'):
            return self._dlos_duv1

        (dlos_du, dlos_dv) = self.dlos_duv.extract_denoms()

        self._dlos_duv1 = self.dlos_duv.copy()
        (dlos_du1, dlos_dv1) = self._dlos_duv1.extract_denoms()
            # memory is shared between self._dlos_duv1, dlos_du1, and dlos_dv1
            # so changes to dlos_du1, and dlos_dv1 appear inside self._dlos_duv1

        # Select pixels where the u-size is smaller
        u_is_smaller = dlos_du.norm_sq().vals <= dlos_dv.norm_sq().vals

        # Here, replace v with its component perpendicular to u
        dlos_dv1[u_is_smaller] = dlos_dv.perp(dlos_du)[u_is_smaller]

        # Select pixels where the v is smaller
        v_is_smaller = np.logical_not(u_is_smaller)

        # Here, replace u with its component perpendicular to v
        dlos_du1[v_is_smaller] = dlos_du.perp(dlos_dv)[v_is_smaller]

        return self._dlos_duv1

    @property
    def duv_dlos(self):
        if not hasattr(self, '_duv_dlos'):
            self._duv_dlos = self.meshgrid.duv_dlos(self.time)

        return self._duv_dlos

    @property
    def center_dlos_duv(self):
        if not hasattr(self, '_center_dlos_duv'):
            self._center_dlos_duv = self.meshgrid.center_dlos_duv(self.time)

        return self._center_dlos_duv

    @property
    def center_duv_dlos(self):
        if not hasattr(self, '_center_duv_dlos'):
            self._center_duv_dlos = self.meshgrid.center_duv_dlos(self.time)

        return self._center_duv_dlos

    ############################################################################
    # Dictionary keys
    ############################################################################

    @functools.lru_cache(maxsize=300)
    def standardize_event_key(self, event_key, default=''):
        """Repair an event key to make it suitable for indexing a dictionary.

        The photons originate from the Sun unless otherwise indicated. An empty
        event_key is returned as an empty tuple.

        Use default = 'ANSA', 'RING', or 'LIMB' to add this suffix to the body
        name if no suffix is specified.
        """

        if not event_key:
            return ()

        # Handle an individual string
        if isinstance(event_key, str):
            event_key = ('SUN<', event_key.upper())
        else:
            # Handle a tuple of strings
            event_key = [k.upper() for k in event_key]

        # Begin with the Sun if there is no illumination already
        if event_key[0][-1] not in ('<', '>', '-'):
            event_key = ['SUN<'] + event_key

        # If SUN is duplicated, remove the extra one
        # This happens when re-using old event keys that did not have the suffix
        if event_key[0][:-1] == event_key[1]:
            event_key = event_key[:1] + event_key[2:]

        # Add the default surface suffix to the body if necessary
        if default and ':' not in event_key[-1]:
            default = default.upper()
            surface = self.get_surface(event_key[-1])
            if surface.COORDINATE_TYPE == 'spherical':
                event_key = event_key[:-1] + (event_key[-1] + ':' + default,)
            elif default == 'ANSA' and surface.COORDINATE_TYPE == 'polar':
                event_key = event_key[:-1] + (event_key[-1] + ':' + default,)

        event_key = tuple(event_key)

        # Check length
        if self._is_dispersed(event_key) and len(event_key) not in (2,3):
            raise ValueError('illegal surface event key: ' + repr(event_key))

        if self._is_occultation(event_key) and len(event_key) != 2:
            raise ValueError('illegal occultation event key: '
                             + repr(event_key))

        if self._is_gridless(event_key) and len(event_key) != 2:
            raise ValueError('illegal gridless event key: ' + repr(event_key))

        return event_key

    def _is_dispersed(self, event_key):
        if len(event_key) == 0:
            return True
        return event_key[0][-1] == '<'

    def _is_occultation(self, event_key):
        if len(event_key) == 0:
            return False
        return event_key[0][-1] == '>'

    def _is_gridless(self, event_key):
        if len(event_key) == 0:
            return False
        return event_key[0][-1] == '-'

    def _is_shadowing(self, event_key):
        if len(event_key) == 0:
            return False
        return self._is_dispersed(event_key) and len(event_key) == 3

    def gridless_event_key(self, event_key, default=''):
        """Convert event key to gridless."""

        event_key = self.standardize_event_key(event_key, default=default)
        if not event_key:
            return event_key

        return (event_key[0][:-1] + '-',) + event_key[1:]

    #===========================================================================
    def standardize_backplane_key(self, backplane_key):
        """Repair a backplane key to make it suitable for indexing a dictionary.

        A string is turned into a tuple. Strings are converted to upper case. If
        the argument is a backplane already, the key is extracted from it.
        """

        if isinstance(backplane_key, Qube):
            if hasattr(backplane_key, 'key'):
                return backplane_key.key

            for key, value in self.backplanes.items():
                if value is backplane_key:
                    return key

            raise ValueError('illegal backplane key type: ' +
                              type(backplane_key).__name__)

        return self._standardize_backplane_key_if_not_qube(backplane_key)

    @functools.lru_cache(maxsize=300)
    def _standardize_backplane_key_if_not_qube(self, backplane_key):

        if isinstance(backplane_key, str):
            backplane_key = (backplane_key.upper(),)

        elif isinstance(backplane_key, tuple):
            pass

        else:
            raise ValueError('illegal backplane key type: ' +
                              type(backplane_key).__name__)

        return backplane_key

    #===========================================================================
    @functools.lru_cache(maxsize=300)
    def _event_and_backplane_keys(self, event_key, names=(), default=''):
        """Interpret the input as either a backplane_key or an event_key.

        names is the set of possible backplane names to seek; if not specified,
        the complete set of defined Backplane names is used.

        Use default = 'ANSA', 'RING', or 'LIMB' to add this suffix to the body
        name if no suffix is specified.
        """

        if not names:
            names = Backplane.CALLABLES

        backplane_key = event_key
        uses_backplane_key = False
        while event_key[0] in names:
            event_key = event_key[1]
            uses_backplane_key = True

        event_key = self.standardize_event_key(event_key, default=default)
        if uses_backplane_key:
            backplane_key = self.standardize_backplane_key(backplane_key)
            return (event_key, backplane_key)

        return (event_key, None)

    #===========================================================================
    @functools.lru_cache(maxsize=300)
    def get_body_and_modifier(self, surface_key):
        """A body object and modifier based on the given surface key.

        The string is normally a registered body ID (case insensitive), but it
        can be modified with ':ANSA', ':RING' or ':LIMB' to indicate an
        associated surface.
        """

        surface_id = surface_key.upper()

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

    #===========================================================================
    @functools.lru_cache(maxsize=300)
    def unmasked_surface_key(self, surface_key):
        """The unmasked surface key associated with a given surface key.
        Example: SATURN_MAIN_RINGS -> SATURN:RING.

        If the surface has no associated unmasked surface, the same surface key
        is returned.
        """

        (body, modifier) = self.get_body_and_modifier(surface_key)

        # If this is an ansa surface, make sure the parent is a planet, not ring
        if modifier == 'ANSA':
            if body.surface.COORDINATE_TYPE == 'polar':
                body = body.parent

        body_name = body.name.upper()

        # If the string has a modifier, it must be the default surface
        if modifier:
            return body_name + ':' + modifier

        # Determine what kind of a surface this is
        intercept_key = self.get_surface(surface_key).intercept_key
        surface_type = intercept_key[0]

        if surface_type == 'ellipsoid':
            return body_name

        parent = body.parent

        # If it's a ring or orbit, we must confirm that it has zero inclination
        # or elevation, and uses the parent body's ring_frame. Format is:
        # ('ring', origin, frame, elevation, i, node, dnode_dt, epoch)
        if surface_type == 'ring':
            if intercept_key[3:] != (0., 0., 0., 0., 0.):
                return surface_key
            if intercept_key[2] is not parent.ring_frame.wayframe:
                return surface_key

            return body_name + ':RING'

        # ('ansa', origin, frame)
        if surface_type == 'ansa':
            if intercept_key[2] is not parent.ring_frame:
                return surface_key

            return body_name + ':ANSA'

        return ''

    #===========================================================================
    @functools.lru_cache(maxsize=300)
    def unmasked_event_key(self, event_key):
        """Return the unmasked event key based on an event key.
        """

        event_key = self.standardize_event_key(event_key)

        new_key = [event_key[0]]
        for surface_key in event_key[1:]:
            new_key.append(self.unmasked_surface_key(surface_key))

        return tuple(new_key)

    #===========================================================================
    @functools.lru_cache(maxsize=300)
    def intercept_dict_key(self, event_key):
        """Return the key for the intercepts dictionary based on an event key.
        """

        event_key = self.standardize_event_key(event_key)

        new_key = [event_key[0]]
        for surface_key in event_key[1:]:
            surface = self.get_surface(surface_key)
            new_key.append(surface.intercept_key)

        return tuple(new_key)

    ############################################################################
    # Event solver
    ############################################################################

    def get_obs_event(self, event_key, derivs=False):
        """The observation event of photons arriving at the detector."""

        derivs = derivs or self.ALL_DERIVS

        # Gridded events always carry the same arrival vectors
        if self._is_dispersed(event_key):
            return self.obs_events[derivs]

        # For others, check the target-based cache first
        if event_key in self.gridless_arrivals:
            event = self.gridless_arrivals[event_key]
            return (event if derivs else event.wod)

        # Occultation events depend on the source
        if self._is_occultation(event_key):
            source_key = event_key[0][:-1]
        else:
            # Gridless events depend on the last body
            source_key = event_key[-1]

        # Check the cache
        if source_key in self.gridless_arrivals:
            arrival = self.gridless_arrivals[source_key]
            departure = None

        # Otherwise, solve
        else:
            source = self.get_body_and_modifier(source_key)[0]
            (departure,
             arrival) = source.photon_to_event(self.obs_gridless_event,
                                               derivs=True)
            self.gridless_arrivals[source_key] = arrival

        # Save in the arrivals dictionary for next time
        self.gridless_arrivals[event_key] = arrival

        # Save the departure event if any
        if departure is not None and event_key not in self.surface_events[True]:
            surface = self.get_surface(event_key[-1])
            departure = departure.wrt(surface.origin, surface.frame)
            self.surface_events[True ][event_key] = departure
            self.surface_events[False][event_key] = departure.wod

        return arrival

    #===========================================================================
    @functools.lru_cache()
    def get_surface(self, surface_key):
        """A surface based on its surface key."""

        (body, modifier) = self.get_body_and_modifier(surface_key)

        if modifier is None:
            return body.surface

        if modifier == 'RING':
            if body.ring_body is not None:
                return body.ring_body.surface

            return RingPlane(body.path, body.ring_frame, gravity=body.gravity)

        if modifier == 'ANSA':
            if body.surface.COORDINATE_TYPE == 'polar':     # if it's a ring
                return Ansa.for_ringplane(body.surface)
            else:                                           # if it's a planet
                return Ansa(body.path, body.ring_frame)

        if modifier == 'LIMB':
            return Limb(body.surface)

    #===========================================================================
    def get_antimask(self, surface_key):
        """Prepare a rectangular antimask for a particular surface event.

        The antimask defines the bounding box of the meshgrid that intercepts
        the given surface.
        """

        # Return from the antimask cache if present
        if surface_key in self.antimasks:
            return self.antimasks[surface_key]

        # If the inventory is disabled, we're done
        if self.inventory is None:
            return True

        # For a name with a colon, there is no antimask
        if ':' in surface_key:
            return True

        # Update the inventory if necessary
        body_name = surface_key
        if body_name not in self.inventory:
            self.inventory.update(self.obs.inventory([body_name],
                                                     return_type='full'))

        # If it is absent from the inventory now, it's not in the image
        if body_name not in self.inventory:
            self.antimasks[body_name] = False
            return False

        body_dict = self.inventory[body_name]
        if not body_dict['inside']:
            self.antimasks[body_name] = False
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

        self.antimasks[body_name] = antimask
        return antimask

    #===========================================================================
    def get_surface_event(self, event_key, derivs=False, arrivals=False):
        """The surface or path event surface based on an event key.

        Inputs:
            event_key       event key.
            derivs          True for events with time and LOS derivs; False
                            otherwise.
            arrivals        True for an event requiring arrival vectors; False
                            otherwise.
        """

        derivs = derivs or self.ALL_DERIVS

        # Handle the empty event key (used for sky coordinates) quickly
        event_key = self.standardize_event_key(event_key)
        if not event_key:
            return self.get_obs_event(event_key, derivs)

        # Retrieve the event from the cache if it is available
        if event_key in self.surface_events[derivs]:
            event = self.surface_events[derivs][event_key]

            # Fill in the arrivals if needed
            if arrivals and not event.has_arrivals():
                source = Body.lookup(event_key[0][:-1])
                event = source.photon_to_event(event,
                                               antimask=event.antimask,
                                               derivs=derivs)[1]
                surface = self.get_surface(event_key[1])
                event = event.wrt(surface.origin, surface.frame)
                self._save_event(event_key, event, surface=surface,
                                                   derivs=derivs)

            # Fill in the perpendicular if needed
            if event.perp is None:
                event.perp = Vector3.ZAXIS
                self.surface_events[derivs][event_key] = event

            return event

        # Always include derivatives by default, except for shadowing events
        is_shadowing   = self._is_shadowing(event_key)
        is_occultation = self._is_occultation(event_key)
        is_gridless    = self._is_gridless(event_key)

        if not is_shadowing and not derivs:
            try:
                event = self.get_surface_event(event_key, derivs=True,
                                                          arrivals=arrivals)
                return event.wod
            except NotImplementedError:
                pass

        # Get the detection event at the next surface (or at the detector)
        # For shadowing events, this is the event associated with the two-item
        # key.
        if is_shadowing:
            two_item_key = event_key[::2]       # skip over middle item
            detection = self.get_surface_event(two_item_key, derivs=derivs,
                                                             arrivals=True)

        # For occultation events, the line of sight from the detector is
        # defined by the source.
        elif is_occultation:
            source = Body.lookup(event_key[0][:-1])
            if source in self.obs_occultation_events:
                detection = self.obs_occultation_events[source]
            else:
                detection = source.photon_to_event(self.obs_gridless_event,
                                                   derivs=True)
                detection = event.wrt_frame(self.obs.frame)
                self.obs_occultation_events[source] = detection

        else:
            detection = self.get_obs_event(event_key, derivs=derivs)

        # Solve for the event
        if is_gridless:
            # If gridless, the call to get_obs_event above filled in the event,
            # but fill in the perpendicular if necessary
            event = self.surface_events[derivs][event_key]
            if event.perp is None:
                event.perp = Vector3.ZAXIS
                self.surface_events[derivs][event_key] = event

        else:   # is_dispersed or is_occultation
            event = self._get_los_event(event_key, detection, derivs)

        # Fill in the arrivals recursively if needed
        if arrivals and not event.has_arrivals():
            event = self.get_surface_event(event_key, derivs=derivs,
                                                      arrivals=arrivals)

        return event

    #===========================================================================
    def _get_los_event(self, event_key, detection, derivs):
        """The event defined by dispersed lighting over a surface."""

        surface_key = event_key[1]
        surface = self.get_surface(surface_key)

        if self._is_dispersed(event_key):
            antimask = self.get_antimask(event_key[-1]) # last item defines mask
        else:
            antimask = None

        # Update the intercept dictionary if necessary
        intercept_key = self.intercept_dict_key(event_key)
        unmasked_event_key = self.unmasked_event_key(event_key)

        if intercept_key in self.intercepts[derivs]:
            event = self.intercepts[derivs][intercept_key]
            if self.DIAGNOSTICS:
                LOGGING.diagnostic('INTERCEPT REUSED', event_key,
                                   'derivs=%s' % derivs)

        else:
            now = datetime.datetime.now()
            event = surface.unmasked.photon_to_event(detection,
                                                     antimask=antimask,
                                                     derivs=derivs)[0]
            if self.PERFORMANCE:
                elapsed = (datetime.datetime.now() - now).total_seconds()
                LOGGING.performance('INTERCEPT %6.3f' % elapsed, event_key,
                                    'derivs=%s' % derivs)

            # Save in the intercepts dictionary
            self.intercepts[derivs][intercept_key] = event
            if derivs:
                self.intercepts[False][intercept_key] = event.wod

            # Also save the unmasked event in the surface dictionary
            self._save_event(unmasked_event_key, event, surface=surface,
                                                        derivs=derivs)

        # Apply the coordinates and mask
        if event_key != unmasked_event_key:
            event = surface.apply_coords_to_event(
                                    event,
                                    obs=self.get_obs_event(event_key, derivs))
            self._save_event(event_key, event, surface=surface, derivs=derivs)

        return event

    #===========================================================================
    def _save_event(self, event_key, event, surface, derivs):

        body = self.get_body_and_modifier(event_key[1])[0]
        event.insert_subfield('body', body)
        event.insert_subfield('event_key', event_key)
        event.insert_subfield('surface', surface)

        # Save the event
        derivs = derivs or self.ALL_DERIVS
        self.surface_events[derivs][event_key] = event

        # Save the antimask
        if self._is_dispersed(event_key) and not self._is_shadowing(event_key):
            surface_key = event_key[-1]
            if surface_key not in self.antimasks:
                self.antimasks[surface_key] = event.antimask

        # Save the without-derivs version if necessary
        if derivs:
            self.surface_events[False][event_key] = event.wod

    #===========================================================================
    def get_gridless_event(self, event_key, derivs=False, arrivals=False):
        """The gridless event associated with this event key, even if the event
        key refers to dispersed or occultation lighting.
        """

        derivs = derivs or self.ALL_DERIVS

        gridless_key = self.gridless_event_key(event_key)
        return self.get_surface_event(gridless_key, derivs=derivs,
                                                    arrivals=arrivals)

    ############################################################################
    # Backplane support
    ############################################################################

    def register_backplane(self, key, backplane, expand=False, derivs=False):
        """Insert this backplane into the dictionary.

        If expand is True and the backplane contains just a single value, the
        backplane is expanded to the overall shape.
        """

        if isinstance(backplane, (np.bool_, bool)):
            backplane = Boolean(bool(backplane))

        elif isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)

        # Collapse mask if possible
        backplane = backplane.collapse_mask()

        # Under some circumstances a derived backplane can be a scalar
        if expand and backplane.shape == () and self.shape != ():
            if isinstance(backplane, Boolean):
                vals = np.empty(self.shape, dtype='bool')
                vals[...] = backplane.vals
                backplane = Boolean(vals, backplane.mask)
            else:
                vals = np.empty(self.shape, dtype='float')
                vals[...] = backplane.vals
                backplane = Scalar(vals, backplane.mask)

        # For reference, we add the key as an attribute of each backplane
        # object
        backplane.key = key
        backplane = backplane.as_readonly(recursive=True)
        self.backplanes[key] = backplane.wod

        if backplane.derivs:
            self.backplanes_with_derivs[key] = backplane

        if derivs or self.ALL_DERIVS:
            return backplane
        else:
            return backplane.wod

    #===========================================================================
    def _remasked_backplane(self, key, backplane_key, derivs=False):
        """Apply the mask of one backplane to another."""

        derivs = derivs or self.ALL_DERIVS

        array = self.evaluate(key, derivs=derivs)
        mask = self.evaluate(backplane_key).mask
        array = array.remask_or(mask, recursive=derivs)

        new_key = key[:1] + (backplane_key,) + key[2:]
        self.register_backplane(new_key, array)
        return array

    #===========================================================================
    def get_backplane(self, key, derivs=False):
        """Return the selected backplane from the cache."""

        if (derivs or self.ALL_DERIVS) and key in self.backplanes_with_derivs:
            return self.backplanes_with_derivs[key]

        return self.backplanes[key]

    ############################################################################
    # Method to access a backplane or mask by key
    ############################################################################

    # Here we use the class introspection capabilities of Python to provide a
    # general way to generate any backplane based on its key. This makes it
    # possible to access any backplane via its key rather than by making an
    # explicit call to the function that generates the key.

    # Here we keep track of all the function names that generate backplanes. For
    # security, we disallow evaluate() to access any function not in this list.

    CALLABLES = set()

    def evaluate(self, backplane_key, derivs=False):
        """Evaluate the backplane array based on the given "backplane_key". A
        backplane_key takes the form of a tuple:
            (function_name, event_key, ...)
        where function_name is the name of any Backplane array method, and
        the remaining items in the tuple are the input arguments to that method,
        starting with the event_key.
        """

        if isinstance(backplane_key, str):
            backplane_key = (backplane_key,)

        func = backplane_key[0]
        if func not in Backplane.CALLABLES:
            raise ValueError('unrecognized backplane function: ' + func)

        # Evaluate...
        backplane = Backplane.__dict__[func].__call__(self, *backplane_key[1:])

        derivs = derivs or self.ALL_DERIVS
        if derivs and backplane_key in self.backplanes_with_derivs:
            backplane = self.backplanes_with_derivs[backplane_key]

        return backplane

    #===========================================================================
    @staticmethod
    def _define_backplane_names(globals_dict):
        """Call at the end of each set of Backplane definitions to load them
        into the registry. Input is globals().copy().
        """

        for key, value in globals_dict.items():
            if isinstance(value, types.FunctionType):
                # Move the function into the Backplane name space
                setattr(Backplane, key, value)

                # If it does not start with underscore, save it in the set of
                # callables.
                if key[0] != '_':
                    Backplane.CALLABLES.add(key)

################################################################################
