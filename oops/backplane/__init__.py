################################################################################
# oops/backplane/__init__.py: Backplane class
################################################################################

import types

import numpy as np
from polymath import Boolean, Qube, Scalar

from oops.body              import Body
from oops.path              import AliasPath
from oops.surface.ansa      import Ansa
from oops.surface.limb      import Limb
from oops.surface.ringplane import RingPlane

#===============================================================================
class Backplane(object):
    """Class that supports the generation and manipulation of sets of backplanes
    with a particular observation.

    intermediate results are cached to speed up calculations.
    """

    #===========================================================================
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

        # Filled in if needed
        self._filled_dlos_duv = None
        self._filled_duv_dlos = None

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
        self.obs_gridless_event = obs.gridless_event(self.meshgrid,
                                                     time=self.time)

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
    # Serialization support
    # For quicker backplane generation after unpickling, we save the various
    # internal buffers, but try to avoid any duplication
    ############################################################################

    def __getstate__(self):

        # Check for duplications between dictionaries
        surface_events = {}
        surface_events_w_derivs = {}
        gridless_events = {}

        for key in self.surface_events:
            # Don't save item keyed by () or item with a derivs version
            if key and key not in surface_events_w_derivs:
                surface_events[key] = self.surface_events[key]

        for key in self.surface_events_w_derivs:
            # Don't save item keyed by ()
            if key:
                surface_events[key] = self.surface_events[key]

        for key in self.gridless_events:
            # Don't save item keyed by ()
            if key:
                gridless_events[key] = self.gridless_events[key]

        return (self.obs, self._input_meshgrid, self.time,
                self._input_inventory, self.inventory_border,
                surface_events, surface_events_w_derivs, self.path_events,
                gridless_events, self.gridless_arrivals)

    def __setstate__(self, state):

        (obs, meshgrid, time, inventory, inventory_border,
         surface_events, surface_events_w_derivs, path_events, gridless_events,
         gridless_arrivals) = state

        self.__init__(obs, meshgrid, time, inventory, inventory_border)
        self.surface_events.update(surface_events)
        self.surface_events_w_derivs.update(surface_events_w_derivs)
        self.path_events = path_events
        self.gridless_events.update(gridless_events)
        self.gridless_arrivals = gridless_arrivals

    ############################################################################
    # Key properties
    ############################################################################

    @property
    def dlos_duv(self):
        if self._filled_dlos_duv == None:
            self._filled_dlos_duv = self.meshgrid.dlos_duv(self.time)

        return self._filled_dlos_duv

    @property
    def duv_dlos(self):
        if self._filled_duv_dlos == None:
            self._filled_duv_dlos = self.meshgrid.duv_dlos(self.time)

        return self._filled_duv_dlos

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
        Solar illumination is added at the front if no lighting is specified.
        """

        # Convert to tuple if necessary
        if isinstance(event_key, str):
            event_key = (event_key.upper(),)

        elif type(event_key) == tuple:
            items = []
            for item in event_key:
                if isinstance(item, str):
                    items.append(item.upper())
                else:
                    items.append(item)

            event_key = tuple(items)

        else:
            raise ValueError('illegal event key type: ' + str(type(event_key)))

        return event_key

    #===========================================================================
    @staticmethod
    def standardize_backplane_key(backplane_key):
        """Repair a backplane key to make it suitable for indexing a dictionary.

        A string is turned into a tuple. Strings are converted to upper case. If
        the argument is a backplane already, the key is extracted from it.
        """

        if isinstance(backplane_key, str):
            return (backplane_key.upper(),)

        elif type(backplane_key) == tuple:
            return backplane_key

        elif isinstance(backplane_key, (Scalar, Boolean)):
            return backplane_key.key

        else:
            raise ValueError('illegal backplane key type: ' +
                              str(type(backplane_key)))

    #===========================================================================
    @staticmethod
    def get_body_and_modifier(event_key):
        """A body object and modifier based on the given surface ID.

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

    #===========================================================================
    def get_surface(self, event_key):
        """A surface based on its ID."""

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

    #===========================================================================
    @staticmethod
    def get_path(path_id):
        """A path based on its ID."""

        return Body.lookup(path_id.upper()).path

    #===========================================================================
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

    #===========================================================================
    def get_surface_event(self, event_key):
        """The photon departure event from a surface based on its key."""

        event_key = self.standardize_event_key(event_key)

        # If the event already exists, return it
        if event_key in self.surface_events:
            return self.surface_events[event_key]

        # The Sun is treated as a path, not a surface, unless it is listed last
        if event_key[0] == 'SUN' and len(event_key) > 1:
            return self.get_path_event(event_key)

        # Look up the photon's departure surface
        surface = self.get_surface(event_key)

        # Calculate derivatives for the first step from the observer,
        # if allowed
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

    #===========================================================================
    def get_surface_event_w_derivs(self, event_key):
        """The photon departure event from a surface including derivatives."""

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

    #===========================================================================
    def get_path_event(self, event_key):
        """The departure event from a specified path."""

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

    #===========================================================================
    def get_surface_event_with_arr(self, event_key):
        """The specified event with arrival photons filled in."""

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

    #===========================================================================
    def get_gridless_event(self, event_key):
        """The gridless event identifying a photon departure from a path."""

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

    #===========================================================================
    def get_gridless_event_with_arr(self, event_key):
        """The gridless event with the arrival photons been filled in."""

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

    #===========================================================================
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

    #===========================================================================
    def mask_as_boolean(self, mask):
        """Convert a mask represented by a single boolean, NumPy array, or Qube
        subclass into an unmasked Boolean.

        A masked value is treated as equivalent to False.
        """

        # Undefined values are treated as False
        if isinstance(mask, Qube):
            return Boolean(mask.values.astype('bool') | mask.mask, False)

        if isinstance(mask, np.ndarray):
            return Boolean(mask)

        # Convert a single value to a full array
        if mask:
            return Boolean(np.ones(self.shape, dtype='bool'))
        else:
            return Boolean(np.zeros(self.shape, dtype='bool'))

    #===========================================================================
    def register_backplane(self, key, backplane, expand=True):
        """Insert this backplane into the dictionary.

        If expand is True and the backplane contains just a single value, the
        backplane is expanded to the overall shape.
        """

        if isinstance(backplane, (np.bool_, bool)):
            backplane = Boolean(bool(backplane))

        elif isinstance(backplane, np.ndarray):
            backplane = Scalar(backplane)

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
        backplane = backplane.wod
        backplane.key = key

        self.backplanes[key] = backplane

    #===========================================================================
    def register_gridless_backplane(self, key, backplane):
        """Insert this backplane into the dictionary.

        Same as register_backplane() but without the expansion of a scalar
        value.
        """

        self.register_backplane(key, backplane, expand=False)

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

    def evaluate(self, backplane_key):
        """Evaluates the backplane or mask based on the given key. Equivalent
        to calling the function directly, but the name of the function is the
        first argument in the tuple passed to the function.
        """

        if isinstance(backplane_key, str):
            backplane_key = (backplane_key,)

        func = backplane_key[0]
        if func not in Backplane.CALLABLES:
            raise ValueError('unrecognized backplane function: ' + func)

        return Backplane.__dict__[func].__call__(self, *backplane_key[1:])

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
# UNIT TESTS
################################################################################

import unittest
import os

import oops.config as config

from polymath                   import Vector3

from oops.config                import ABERRATION
from oops.meshgrid              import Meshgrid
from oops.body                  import Body
from oops.event                 import Event
from oops.surface.ansa          import Ansa

from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
from oops.backplane.exercise_backplanes import exercise_backplanes
from oops.backplane.unittester_support  import Backplane_Settings


UNITTEST_SATURN_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/W1573721822_1.IMG')
UNITTEST_RHEA_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/N1649465464_1.IMG')
UNITTEST_UNDERSAMPLE = 16


#*******************************************************************************
class Test_Backplane_Surfaces(unittest.TestCase):

    OLD_RHEA_SURFACE = None


    #===========================================================================
    def setUp(self):
        global OLD_RHEA_SURFACE

        from oops.surface.ellipsoid  import Ellipsoid

        Body.reset_registry()
        Body.define_solar_system('2000-01-01', '2020-01-01')

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


    #===========================================================================
    def tearDown(self):
        global OLD_RHEA_SURFACE

        config.LOGGING.off()
        config.EVENT_CONFIG.collapse_threshold = 3.
        config.SURFACE_PHOTONS.collapse_threshold = 3.

        # Restore Rhea's shape
        Body.as_body('RHEA').surface = OLD_RHEA_SURFACE

        ABERRATION.old = False


    #===========================================================================
    def runTest(self):

      from oops.surface.centricspheroid import CentricSpheroid
      from oops.surface.graphicspheroid import GraphicSpheroid
      from oops.surface.centricellipsoid import CentricEllipsoid
      from oops.surface.graphicellipsoid import GraphicEllipsoid
      import hosts.cassini.iss as iss

      if Backplane_Settings.EXERCISES_ONLY: 
        return

      for ABERRATION.old in (False, True):

        snap = iss.from_file(UNITTEST_SATURN_FILESPEC, fast_distortion=False)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        #meshgrid = Meshgrid(snap.fov, (512,512))
        uv0 = meshgrid.uv
        bp = Backplane(snap, meshgrid)

        # Actual (ra,dec)
        ra = bp.right_ascension(apparent=False)
        dec = bp.declination(apparent=False)

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        ev.neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec)
        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-9)

        # Apparent (ra,dec)  # test doesn't work for ABERRATION=old
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

        # Rhea tests, with Rhea modified
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



#*******************************************************************************
class Test_Backplane_Borders(unittest.TestCase):


    #===========================================================================
    def runTest(self):
      """ 
      These tests are sensitive to the specific kernel pool used.
      """
      import hosts.cassini.iss as iss

      if Backplane_Settings.EXERCISES_ONLY: 
        return

        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                'cassini/ISS/W1573721822_1.IMG')

        snap = iss.from_file(filespec)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        bp = Backplane(obs, meshgrid, inventory=None)



        # Test border of planet intercepted mask, inside
        mask = bp.where_intercepted('saturn')
        
        test = bp.border_inside(mask)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 961)
        self.assertTrue(total-count == 1047615)

        # Test border of planet intercepted mask, outside
        test = bp.border_outside(mask)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 962)
        self.assertTrue(total-count == 1047614)

        # Test border of ring radius below 100 km
        test = bp.border_below(('ring_radius', 'saturn:ring'), 100.e3)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 1713)
        self.assertTrue(total-count == 1046863)

        # Test border of ring radius atop 100 km
        test = bp.border_atop(('ring_radius', 'saturn:ring'), 100.e3)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 1715)
        self.assertTrue(total-count == 1046861)

        # Test border of ring radius above 100 km
        test = bp.border_above(('ring_radius', 'saturn:ring'), 100.e3)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 1715)
        self.assertTrue(total-count == 1046861)

        # Test border of ring radius above 100 km via evaluate()
        test = bp.evaluate(('border_above', ('ring_radius', 'saturn:ring'), 100.e3))
        count = np.sum(test.vals)
        total = np.size(test.vals)
        self.assertTrue(count == 1715)
        self.assertTrue(total-count == 1046861)



#*******************************************************************************
class Test_Backplane_Empty_Events(unittest.TestCase):


    #===========================================================================
    def runTest(self):
      import hosts.cassini.iss as iss

      if Backplane_Settings.EXERCISES_ONLY: 
        return

        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                'cassini/ISS/W1573721822_1.IMG')

        snap = iss.from_file(filespec)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        bp = Backplane(obs, meshgrid, inventory=None)



        # Test empty mask of planet ring radius below 10 km
        test = bp.where_below(('ring_radius', 'saturn_main_rings'), 10.e3)
        count = np.sum(test.vals)
        total = np.size(test.vals)
        percent = int(count / float(total) * 100. + 0.5)
        self.assertTrue(percent == 0)
        
        # Test empty ring radius for Pluto 
        test = bp.ring_radius('pluto:ring')
        total = np.size(test.mask)
        masked = np.sum(test.mask)
        percent = int(masked / float(total) * 100. + 0.5)
        self.assertTrue(percent == 100)

        # Test empty longitude for Pluto
        test = bp.longitude('pluto')
        total = np.size(test.mask)
        masked = np.sum(test.mask)
        percent = int(masked / float(total) * 100. + 0.5)
        self.assertTrue(percent == 100)

        # Test empty incidence angle for Pluto
        test = bp.incidence_angle('pluto')
        total = np.size(test.mask)
        masked = np.sum(test.mask)
        percent = int(masked / float(total) * 100. + 0.5)
        self.assertTrue(percent == 100)



#*******************************************************************************
class Test_Backplane_Exercises(unittest.TestCase):

    #===========================================================================
    def runTest(self):
        import hosts.cassini.iss as iss
        if Backplane_Settings.NO_EXERCISES:
            return 

#        iss.initialize(asof='2019-09-01', mst_pck=True)

        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                'cassini/ISS/W1573721822_1.IMG')

        obs = iss.from_file(filespec)
        bp = exercise_backplanes(obs, 
                                 Backplane_Settings.PRINTING, 
                                 Backplane_Settings.LOGGING, 
                                 Backplane_Settings.SAVING,
                                 Backplane_Settings.OUTPUT, 
                                 undersample=Backplane_Settings.UNDERSAMPLE,
                                 use_inventory=True, inventory_border=4,
                                 planet_key='saturn', 
                                 moon_key='epimetheus', 
                                 ring_key='saturn_main_rings')                                     



##############################################
# See oops/backplane/unittester.py for usage
from oops.backplane.unittester_support      import backplane_unittester_args

if __name__ == '__main__':
    backplane_unittester_args()
    unittest.main(verbosity=2)
################################################################################
