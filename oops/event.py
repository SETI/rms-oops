################################################################################
# oops/event.py: Event class
################################################################################

import sys
import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.transform    import Transform
from oops.config       import EVENT_CONFIG, LOGGING, ABERRATION
from oops.constants    import *

class Event(object):
    """An Event is defined by a time, position and velocity. It always has these
    attributes:

        __time_     event time as a Scalar of arbitrary shape. Times are
                    measured in seconds TDB relative to noon TDB on January 1,
                    2000, consistent with the time system used by the SPICE
                    toolkit.

        __state_    position of the event as a Vector3 of arbitrary shape.
                    Positions are measured in km relative to the specified path.
                    Velocities are specified in km/s as the "t" derivative of
                    the position.

        __origin_   a path object defining the location relative to which all
                    positions and velocities are measured.

        __frame_    a frame object defining the coordinate system in which the
                    components of the positions and velocities are defined.

        __subfields_ an arbitrary dictionary of objects providing further
                     information about the properties of the event.

    It also has these optional attributes:

        __arr_      an optional Vector3 defining the direction of a photon
                    arriving at this event. It is defined in the coordinate
                    frame of this event and not corrected for stellar
                    aberration. Length is arbitrary.

        __arr_ap_   same as above, but defined as the apparent direction of the
                    arriving photons.

        __arr_j2000_
                    same as __arr_, but in J2000 coordinates.

        __arr_ap_j2000_
                    same as __arr_ap_, but in J2000 coordinates.

        __neg_arr_  negative of __arr_ (because it is used so often).

        __neg_arr_ap_
                    negative of __arr_ap_

        __neg_arr_j2000_
                    negative of __arr_j2000_.

        __neg_arr_ap_j2000_
                    negative of __arr_ap_j2000_.

        __arr_lt_   an optional Scalar defining the (negative) light travel time
                    for the arriving photon from its origin.

        __dep_      an optional Vector3 defining the direction of a photon
                    departing from this event. It is defined in the coordinate
                    frame of this event and not corrected for stellar
                    aberration. Length is arbitrary.

        __dep_ap_   same as above, but defined as the apparent direction of the
                    departing photons.

        __dep_j2000_
                    same as __dep_, but in J2000 coordinates.

        __dep_j2000_ap_
                    same as __dep_ap_, but in J2000 coordinates.

        __dep_lt_   an optional Scalar defining the light travel time of a
                    departing photon to its destination.

        __perp_     the direction of a normal vector if this event falls on a
                    surface.

        __vflat_    a velocity component within the surface, which can be used
                    to describe winds across a planet or orbital motion
                    within a ring plane.

        __ssb_      this event referenced to SSB/J2000.

        __xform_to_j2000_
                    transform that converts coordinates in this event to J2000.

    Each of these attributes can also be accessed via a property with the same
    name except for the surrounding undercores.

    Events are intended to be immutable. The exception is that the optional
    attributes can be set exactly once after the constructor is called. You can
    define the photon directions with either apparent or actual values, but not
    both; whichever you define, the other will be generated as needed.

    Note that the attributes of an object need not have the same shape, but they
    must all be broadcastable to the same shape.
    """

    SPECIAL_PROPERTIES = set([
       'arr', 'arr_ap', 'arr_j2000', 'arr_ap_j2000', 'arr_lt',
       'dep', 'dep_ap', 'dep_j2000', 'dep_j2000_ap', 'dep_lt',
       'neg_arr', 'neg_arr_ap', 'neg_arr_j2000', 'neg_arr_ap_j2000',
       'perp', 'vflat'
    ])

    @staticmethod
    def attr_name(prop_name):
        return '_Event__' + prop_name + '_'

    def get_prop(self, prop_name):
        return Event.__dict__[prop_name].fget(self)

    def set_prop(self, prop_name, value):
        Event.__dict__[prop_name].fset(self, value)

    MORE_ATTRIBUTES = set([
        '_Event__ssb_', '_Event__shape_', '_Event__mask_', '_Event__antimask_',
        '_Event__xform_to_j2000_',
    ])

    ############################################################################
    # Avoid circular dependencies with Path class at load time
    SAVED_PATH_CLASS = None

    @classmethod
    def PATH_CLASS(cls):
        if cls.SAVED_PATH_CLASS: return cls.SAVED_PATH_CLASS
        if 'oops.path_.path' in sys.modules:
            cls.SAVED_PATH_CLASS = sys.modules['oops.path_.path'].Path
        return cls.SAVED_PATH_CLASS
    ############################################################################

    PACKRAT_ARGS = ['_Event__time_', '_Event__state_', '_Event__origin_',
                    '_Event__frame_',
                    '**_Event__subfields_',
                    '+_Event__arr_',
                    '+_Event__arr_lt_',
                    '+_Event__dep_',
                    '+_Event__dep_lt_',
                    '+_Event__perp_',
                    '+_Event__vflat_',
                    '+_Event__ssb_',
                    '+_Event__shape_',
                    '+_Event__mask_',
                    '+_Event__xform_to_j2000_',
                    '+event_key', '+surface', '+body']

    def __init__(self, time, state, origin, frame=None, **more):
        """Constructor for the Event class

        If a link is specified, then either the arriving or the departing
        photons will be filled in automatically.

        Input:
            time        a Scalar of event times in seconds TDB.
            state       position vectors as a Vector3 object. The velocity
                        should be included as the time-derivative. However, if
                        specified as a tuple of two objects, the first is
                        interpreted as the position and the second as the
                        velocity.
            origin      the path or path ID identifying the origin of this
                        event.
            frame       the frame or frame ID identifying the coordinate frame
                        of this event. By default, it matches the default frame
                        of the origin.
            **more      an arbitrary set of properties and subfields that are
                        will also be accessible as attributes of the Event
                        object. Properties have fixed names and purposes;
                        subfields can be anything.
        """

        self.__time_ = Scalar.as_scalar(time).as_readonly()

        if type(state) in (tuple,list) and len(state) == 2:
            pos = Vector3.as_vector3(state[0])
            vel = Vector3.as_vector3(state[1])
            state = pos.with_deriv('t', vel, 'insert')
        else:
            state = Vector3.as_vector3(state)
            if 't' not in state.derivs:
                state = state.with_deriv('t', Vector3.ZERO)

        self.__state_ = state.as_readonly()

        assert 't' in self.__state_.derivs

        self.__pos_ = self.__state_.without_deriv('t')

        self.__origin_ = Event.PATH_CLASS().as_waypoint(origin)
        self.__frame_ = Frame.as_wayframe(frame) or origin.frame

        # Set default values for properties
        for prop_name in Event.SPECIAL_PROPERTIES:
            self.__dict__[Event.attr_name(prop_name)] = None

        # Initialize additional attributes
        for attr in Event.MORE_ATTRIBUTES:
            self.__dict__[attr] = None

        # Fill in any given subfields or properties
        self.__subfields_ = {}
        for (name, value) in more.iteritems():
            self.insert_subfield(name, value)

    ############################################################################
    # Read-only properties
    ############################################################################

    @property
    def time(self):
        return self.__time_

    @property
    def state(self):
        return self.__state_

    @property
    def pos(self):
        return self.__pos_

    @property
    def vel(self):
        if hasattr(self.__state_, 'd_dt'):
            return self.__state_.d_dt
        else:
            return Vector3.ZERO

    @property
    def origin(self):
        return self.__origin_

    @property
    def origin_id(self):
        return self.__origin_.path_id

    @property
    def frame(self):
        return self.__frame_

    @property
    def frame_id(self):
        return self.__frame_.frame_id

    @property
    def subfields(self):
        return self.__subfields_

    @property
    def shape(self):
        if self.__shape_ is None:
            self.__shape_ = Qube.broadcasted_shape(self.__time_,
                                                   self.__state_,
                                                   self.__origin_,
                                                   self.__frame_)
        return self.__shape_

    @property
    def mask(self):
        if self.__mask_ is None:
            self.__mask_ = (self.__time_.mask | self.__state_.mask |
                                                self.vel.mask)

        return self.__mask_

    @property
    def antimask(self):
        if self.__antimask_ is None:
            self.__antimask_ = np.logical_not(self.mask)

        return self.__antimask_

    @property
    def ssb(self):
        if self.__ssb_ is None:
            self.__ssb_ = self.wrt_ssb(derivs=True)
            self.__ssb_.__ssb_ = self.__ssb_
    
        return self.__ssb_

    @property
    def xform_to_j2000(self):
        """Transform that rotates from event coordinates to J2000 coordinates.
        """

        if self.__ssb_ is None:
            self.__ssb_ = self.wrt_ssb(derivs=True)
            self.__ssb_.__ssb_ = self.__ssb_

        elif self.__xform_to_j2000_ is None:
            frame = Frame.J2000.wrt(self.__frame_)
            self.__xform_to_j2000_ = frame.transform_at_time(self.__time_)

        return self.__xform_to_j2000_

    ############################################################################
    # Special properties: Photon arrival vectors
    ############################################################################

    @property
    def arr(self):
        if self.__arr_ is None:
          if self.__arr_ap_ is not None:
            ignore = self.actual_arr(derivs=True)   # fill internal attribute

        return self.__arr_                  # returns None if still undefined

    @arr.setter
    def arr(self, value):
        if (self.__arr_ is not None) or (self.__arr_ap_ is not None):
            raise ValueError('arriving photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        arr = Vector3.as_vector3(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, arr)

        self.__arr_ = arr
        if ABERRATION.old: self.__arr_ap_ = arr

        if (self.__ssb_ is not None) and (self.__ssb_.__arr_ is None):
            self.__ssb_.__arr_ = self.xform_to_j2000.rotate(self.__arr_)

    @property
    def arr_ap(self):
        if ABERRATION.old: return self.__arr_

        if self.__arr_ap_ is None:
          if self.__arr_ is not None:
            ignore = self.apparent_arr(derivs=True) # fill internal attribute

        return self.__arr_ap_               # returns None if still undefined

    @arr_ap.setter
    def arr_ap(self, value):
        if ABERRATION.old:
            self.arr = value
            return

        if (self.__arr_ap_ is not None) or (self.__arr_ is not None):
            raise ValueError('arriving photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        arr_ap = Vector3.as_vector3(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, arr_ap)

        self.__arr_ap_ = arr_ap

        if (self.__ssb_ is not None) and (self.__ssb_.__arr_ap_ is None):
            self.__ssb_.__arr_ap_ = self.xform_to_j2000.rotate(self.__arr_ap_)

    @property
    def arr_j2000(self):
        return self.ssb.arr

    @arr_j2000.setter
    def arr_j2000(self, value):
        ssb_event = self.ssb

        if self is ssb_event:
            self.arr = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.arr = self.xform_to_j2000.unrotate(value)
            ssb_event.__arr_ = value

    @property
    def arr_ap_j2000(self):
        if ABERRATION.old: return self.arr_j2000

        return self.ssb.arr_ap

    @arr_ap_j2000.setter
    def arr_ap_j2000(self, value):
        if ABERRATION.old:
            self.arr_j2000 = value
            return

        ssb_event = self.ssb

        if self is ssb_event:
            self.arr_ap = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.arr_ap = self.xform_to_j2000.unrotate(value)
            ssb_event.__arr_ap_ = value

    @property
    def arr_lt(self):
        return self.__arr_lt_               # returns None if still undefined

    @arr_lt.setter
    def arr_lt(self, value):
        if self.__arr_lt_ is not None:
            raise ValueError('arriving photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        arr_lt = Scalar.as_scalar(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, arr_lt)

        self.__arr_lt_ = arr_lt

        if (self.__ssb_ is not None) and (self.__ssb_.__arr_lt_ is None):
            self.__ssb_.__arr_lt_ = self.__arr_lt_

    ############################################################################
    # Special properties: Photon departure vectors
    ############################################################################

    @property
    def dep(self):
        if self.__dep_ is None:
          if self.__dep_ap_ is not None:
            ignore = self.actual_dep(derivs=True)   # fill internal attribute

        return self.__dep_                  # returns None if still undefined

    @dep.setter
    def dep(self, value):
        if (self.__dep_ is not None) or (self.__dep_ap_ is not None):
            raise ValueError('departing photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        dep = Vector3.as_vector3(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, dep)

        self.__dep_ = dep
        if ABERRATION.old: self.__dep_ap_ = dep

        if (self.__ssb_ is not None) and (self.__ssb_.__dep_ is None):
            self.__ssb_.__dep_ = self.xform_to_j2000.rotate(self.__dep_)

    @property
    def dep_ap(self):
        if ABERRATION.old: return self.__dep_

        if self.__dep_ap_ is None:
          if self.__dep_ is not None:
            ignore = self.apparent_dep(derivs=True) # fill internal attribute

        return self.__dep_ap_

    @dep_ap.setter
    def dep_ap(self, value):
        if ABERRATION.old: return self.__dep_

        if (self.__dep_ap_ is not None) or (self.__dep_ is not None):
            raise ValueError('departing photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        dep_ap = Vector3.as_vector3(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, dep_ap)

        self.__dep_ap_ = dep_ap

        if (self.__ssb_ is not None) and (self.__ssb_.__dep_ap_ is None):
            self.__ssb_.__dep_ap_ = self.xform_to_j2000.rotate(self.__dep_ap_)

    @property
    def dep_j2000(self):
        return self.ssb.dep

    @dep_j2000.setter
    def dep_j2000(self, value):
        ssb_event = self.ssb

        if self is ssb_event:
            self.dep = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.dep = self.xform_to_j2000.unrotate(value)
            ssb_event.__dep_ = value

    @property
    def dep_ap_j2000(self):
        if ABERRATION.old: return self.dep_j2000

        return self.ssb.dep_ap

    @dep_ap_j2000.setter
    def dep_ap_j2000(self, value):
        if ABERRATION.old:
            self.dep_j2000 = value
            return

        ssb_event = self.ssb

        if self is ssb_event:
            self.dep_ap = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.dep_ap = self.xform_to_j2000.unrotate(value)
            ssb_event.__dep_ap_ = value

    @property
    def dep_lt(self):
        return self.__dep_lt_

    @dep_lt.setter
    def dep_lt(self, value):
        if self.__dep_lt_ is not None:
            raise ValueError('departing photons were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        dep_lt = Scalar.as_scalar(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, dep_lt)

        self.__dep_lt_ = dep_lt

        if (self.__ssb_ is not None):
            self.__ssb_.__dep_lt_ = self.__dep_lt_

    ############################################################################
    # Special properties: Photon arrival vectors, reversed
    ############################################################################

    @property
    def neg_arr(self):
        if self.__neg_arr_ is None:
          self.__neg_arr_ = -self.arr

        return self.__neg_arr_

    @neg_arr.setter
    def neg_arr(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.arr = -value
        self.__neg_arr_ = value

    @property
    def neg_arr_ap(self):
        if ABERRATION.old: return self.neg_arr

        if self.__neg_arr_ap_ is None:
          self.__neg_arr_ap_ = -self.arr_ap

        return self.__neg_arr_ap_

    @neg_arr_ap.setter
    def neg_arr_ap(self, value):
        if ABERRATION.old:
            self.neg_arr = value
            return

        value = Vector3.as_vector3(value).as_readonly()
        self.arr_ap = -value
        self.__neg_arr_ap_ = value

    @property
    def neg_arr_j2000(self):
        return self.ssb.neg_arr

    @neg_arr_j2000.setter
    def neg_arr_j2000(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.ssb.arr = -value
        self.ssb.__neg_arr_ = value

        if self.ssb is not self:
            self.arr = self.xform_to_j2000.unrotate(self.ssb.__arr_)

    @property
    def neg_arr_ap_j2000(self):
        if ABERRATION.old: return self.neg_arr_j2000

        return self.ssb.neg_arr_ap

    @neg_arr_ap_j2000.setter
    def neg_arr_ap_j2000(self, value):
        if ABERRATION.old:
            self.neg_arr_j2000 = value
            return

        value = Vector3.as_vector3(value).as_readonly()
        self.ssb.arr_ap = -value
        self.ssb.__neg_arr_ap_ = value

        if self.ssb is not self:
            self.arr_ap = self.xform_to_j2000.unrotate(self.ssb.__arr_ap_)

    ############################################################################
    # Special properties: Additional surface properties
    ############################################################################

    @property
    def perp(self):
        return self.__perp_

    @perp.setter
    def perp(self, value):
        if self.__perp_ is not None:
            raise ValueError('perpendiculars were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        perp = Vector3.as_vector(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, perp)

        self.__perp_ = perp

        if (self.__ssb_ is not None) and (self.__ssb_.__perp_ is None):
            self.__ssb_.__perp_ = self.xform_to_j2000.rotate(self.__perp_)

    @property
    def vflat(self):
        if self.__vflat_ is None:
            self.__vflat_ = Vector3.ZERO

        return self.__vflat_

    @vflat.setter
    def vflat(self, value):
        if self.__vflat_ is not None:
            raise ValueError('surface velocities were already defined in ' + 
                             str(self))

        # Raise a ValueError if the shape is incompatible
        vflat = Vector3.as_vector(value).as_readonly()
        self.__shape_ = Qube.broadcasted_shape(self.shape, vflat)

        self.__vflat_ = vflat

        if (self.__ssb_ is not None) and (self.__ssb_.__vflat_ is None):
            self.__ssb_.__vflat_ = self.xform_to_j2000.rotate(self.__vflat_)

    ############################################################################
    # Standard methods
    ############################################################################

    def __str__(self):
        time = self.time.flatten()
        pos = self.pos.flatten()
        vel = self.vel.flatten()

        str_list = ['Event(time = ', ]
        if time.shape == ():
            str_list.append(str(time))
        elif time.size == 1:
            str_list.append(str(time[0]))
        elif time.size == 2:
            str_list += [str(time)]
        else:
            str_list += [str(time[0]), ', ..., ', str(time[-1])]

        str_list.append(';\n  pos = ')
        if pos.shape == ():
            str_list.append(str(pos))
        elif pos.size == 1:
            str_list.append(str(pos[0]))
        elif pos.size == 2:
            str_list += [str(pos)]
        else:
            str_list += [str(pos[0]), ', ..., ', str(pos[-1])]

        str_list.append(';\n  vel = ')
        if vel.shape == ():
            str_list.append(str(vel))
        elif vel.size == 1:
            str_list.append(str(vel[0]))
        elif vel.size == 2:
            str_list += [str(vel)]
        else:
            str_list += [str(vel[0]), ', ..., ', str(vel[-1])]

        str_list += [';\n  shape = ', str(self.shape), ', ',
                     self.__origin_.path_id, ', ',
                     self.__frame_.frame_id]

        keys = self.__subfields_.keys()
        keys.sort()
        for key in keys:
            str_list += ['; ', key]

        str_list += [')']
        return ''.join(str_list)

    ############################################################################
    # Subfield and property methods
    ############################################################################

    def insert_subfield(self, name, value):
        """Insert a given subfield into this Event."""

        if name in Event.SPECIAL_PROPERTIES:
            self.set_prop(name, value)
        else:
            self.__dict__[name] = value
            self.__subfields_[name] = value

            if self.__ssb_ is not None and self.__ssb_ is not self:
                try:
                    rotated = self.xform_to_j2000.unrotate(value)
                except:
                    rotated = value

                self.__ssb_.insert_subfield(name, rotated)

    def get_subfield(self, name):
        """Return the value of a given subfield or property."""

        if name in Event.SPECIAL_PROPERTIES:
            return self.get_prop(name)

        return self.subfields[name]

    ############################################################################
    # Constructors for event variants
    ############################################################################

    def clone(self, recursive=False):
        """A shallow copy of the Event.

        Inputs:
            recursive   True also to clone (shallow-copy) the attributes of the
                        Event. This is necessary if derivatives of the subfields
                        are going to be modified.
        """

        def clone_attr(arg):
            if recursive:
                try:
                    return arg.clone()
                except:
                    return arg
            else:
                return arg

        result = Event(clone_attr(self.__time_),
                       clone_attr(self.__state_),
                       self.__origin_, self.__frame_)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = clone_attr(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = self.__dict__[attr]

        if result.__ssb_ == self:   # This needs to point to self if possible
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.clone(recursive=recursive)
            result.__ssb_.__ssb_ = result.__ssb_

        for (name, value) in self.__subfields_.iteritems():
            result.insert_subfield(name, clone_attr(value))

        return result

    #### Does this handle time dilation correctly??
    def with_time_derivs(self):
        """Return a clone of this event containing unit time derivatives d_dt
        in the frame of the event.
        """

        if 't' in self.__time_.derivs: return self

        event = self.clone(recursive=True)
        event.__time_.insert_deriv('t', Scalar.ONE, override=True)

        #### Does this handle time dilation correctly??
        if event.__ssb_ is not None and event.__ssb_ is not event:
            event.ssb.__time_.insert_deriv('t', Scalar.ONE, override=True)

        return event

    def with_los_derivs(self):
        """Clone of this event with unit photon arrival derivatives d_dlos.
        """

        if 'los' in self.neg_arr_ap.derivs: return self

        event = self.clone(recursive=True)

        event.neg_arr_ap.insert_deriv('los', Vector3.IDENTITY, override=True)
        event.arr_ap.insert_deriv('los', -Vector3.IDENTITY, override=True)

        if not ABERRATION.old: event.__arr_ = None

        if event.ssb is not None and event.ssb is not event and \
           event.ssb.__arr_ap_ is not None:
            event.ssb.__arr_ap_ = event.xform_to_j2000.rotate(event.__arr_ap_,
                                                              derivs=True)
            event.ssb.__neg_arr_ap_ = None

            if not ABERRATION.old: event.ssb.__arr_ = None

        return event

    #### Needs validation
    def with_pos_derivs(self):
        """Return a clone of this event containing unit position derivatives
        d_dpos in the frame of the event.
        """

        if 'pos' in event.__state__.derivs: return self

        event = self.clone(recursive=True)
        event.__state_.insert_deriv('pos', Vector3.IDENTITY, override=True)

        if event.__ssb_ is not None and event.__ssb_ is not event:
            dpos_dpos_j2000 = event.xform_to_j2000.rotate(event.__state_,
                                                          derivs=True)
            event.ssb.__state_.insert_deriv('pos', Vector3.IDENTITY,
                                                   override=True)

        return event

    #### Does this handle time dilation correctly??
    def with_lt_derivs(self):
        """Return a clone of this event containing unit photon arrival
        light-time derivatives d_dlt.
        """

        if 'lt' in self.arr_lt.derivs: return self

        event = self.clone(recursive=True)
        event.__arr_lt_.insert_deriv('lt', Scalar.ONE, override=True)

        if event.__ssb_ is not None and event.__ssb_ is not event:
            event.ssb.arr_lt.insert_deriv('lt', Scalar.ONE, override=True)

        return event

    #### Does this handle time dilation and aberration correctly??
    def with_dep_derivs(self):
        """Clone of this event with unit photon departure derivatives d_ddep.
        """

        if 'dep' in self.dep_ap.derivs: return self

        event = self.clone(recursive=True)
        event.dep_ap.insert_deriv('dep', Vector3.IDENTITY, override=True)

        if not ABERRATION.old: event.__dep_ = None

        if event.ssb is not None and event.ssb is not event and \
           event.ssb.__dep_ap_ is not None:
            event.ssb.__dep_ap_ = event.xform_to_j2000.rotate(event.__dep_ap_,
                                                              derivs=True)

            if not ABERRATION.old: event.ssb.__dep_ = None

        return event

    #### Does this handle time dilation correctly??
    def with_dlt_derivs(self):
        """Return a clone of this event containing unit photon departure
        light-time derivatives d_ddlt.
        """

        if 'dlt' in self.dep_lt.derivs: return self

        event = self.clone(recursive=True)
        event.__dep_lt_.insert_deriv('dlt', Scalar.ONE, override=True)

        if event.__ssb_ is not None and event.__ssb_ is not event:
            event.ssb.dep_lt.insert_deriv('dlt', Scalar.ONE, override=True)

        return event

    def without_derivs(self):
        """A shallow copy of this Event without any derivatives except time.
        """

        def remove_derivs(arg):
            try:
                return arg.without_derivs(preserve='t')
            except:
                return arg

        result = Event(remove_derivs(self.__time_),
                       remove_derivs(self.__state_),
                       self.__origin_, self.__frame_)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = remove_derivs(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = self.__dict__[attr]

        if self.__ssb_ == self:     # This needs to point to self if possible
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.without_derivs()
            result.__ssb_._ssb_ = result.__ssb_

        for (name, value) in self.__subfields_.iteritems():
            result.insert_subfield(name, remove_derivs(value))

        return result

    def without_mask(self):
        """A shallow copy of this Event with the masks removed."""

        def remove_mask(arg):
            try:
                return arg.without_mask()
            except:
                return arg

        result = Event(remove_mask(self.__time_),
                       remove_mask(self.__state_),
                       self.__origin_, self.__frame_)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = remove_mask(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = self.__dict__[attr]

        if self.__ssb_ == self:     # This needs to point to self if possible
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.remove_mask()
            result.__ssb_._ssb_ = result.__ssb_

        self.__mask_ = False

        for (name, value) in self.__subfields_.iteritems():
            result.insert_subfield(name, remove_mask(value))

        return result

    def all_masked(self, origin=None, frame=None):
        """A shallow copy of this event, entirely masked.

        Inputs:
            origin      the origin or origin_id of the Event returned; if None,
                        use the origin of this Event.
            frame       the frame or frame_id of the Event returned; if None,
                        use the frame of this Event.
        """

        def fully_masked(arg):
            try:
                return arg.all_masked()
            except:
                return arg

        result = Event(self.__time_.all_masked(),
                       self.__state_.all_masked(),
                       self.__origin_, self.__frame_)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = fully_masked(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = self.__dict__[attr]

        if self.__ssb_ is self:
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.all_masked()
            result.__ssb_._ssb_ = result.__ssb_

        self.__mask_ = True

        for (name, value) in self.__subfields_.iteritems():
            result.insert_subfield(name, fully_masked(value))

        return result

    def apply_mask(self, mask):
        """A shallow copy of this Event with a new mask."""

        def apply_mask_to(arg):
            try:
                return arg.mask_where(mask)
            except:
                return arg

        result = Event(self.__time_.mask_where(mask, recursive=True),
                       self.__state_.mask_where(mask, recursive=True),
                       self.__origin_, self.__frame_)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = apply_mask_to(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = self.__dict__[attr]

        if self.__ssb_ is self:
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.apply_mask(mask)
            result.__ssb_._ssb_ = result.__ssb_

        for (name, value) in self.__subfields_.iteritems():
            result.insert_subfield(name, apply_mask_to(value))

        return result

    def replace(self, *args):
        """A shallow copy with one property replaced.
        """

        result = self.clone()

        pairs = []
        for k in range(0,len(args),2):
            pairs.append((args[k],args[k+1]))

        for (name, value) in pairs:
            if name in ('arr_lt', 'dep_lt', 'perp', 'vflat'):
                result.__dict__[Event.attr_name(name)] = None
            elif name[:3] in ('arr', 'neg'):
                result.__arr_ = None
                result.__arr_ap_ = None
                result.__arr_j2000_ = None
                result.__arr_ap_j2000_ = None
                result.__neg_arr_ = None
                result.__neg_arr_j2000_ = None
                result.__neg_arr_ap_ = None
                result.__neg_arr_ap_j2000 = None
            elif name[:3] == 'dep':
                result.__dep_ = None
                result.__del_ap_ = None
                result.__dep_j2000_ = None
                result.__del_ap_j2000_ = None

        for (name, value) in pairs:
            result.insert_subfield(name, value)

        return result

    ############################################################################
    # Shrink and unshrink operations
    ############################################################################

    def shrink(self, antimask):
        """Return a shrunken version of this event.

        Antimask is None to leave the Event unchanged; otherwise, it must be
        True where values are kept and False where they are ignored. A single
        boolean value of True keeps everything; a single boolean value of False
        ignores everything
        """

        def shrink1(arg):
            try:
                return arg.shrink(antimask)
            except AttributeError:
                return arg

        if antimask is None: return self
        if antimask is True: return self
        if antimask is False: return self.all_masked()

        result = Event(self.time.shrink(antimask), self.state.shrink(antimask),
                       self.origin, self.frame)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = shrink1(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            if attr == '_Event__ssb_': continue
            result.__dict__[attr] = shrink1(self.__dict__[attr])

        for (name, value) in self.subfields.iteritems():
            result.insert_subfield(name, shrink1(value))

        if self.__ssb_ is self:
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.shrink(antimask)
            result.__ssb_._ssb_ = result.__ssb_

        result.__shape_ = None

        if result.__xform_to_j2000_ is not None:
            xform = result.__xform_to_j2000_
            new_xform = Transform(xform.matrix.shrink(antimask),
                                  xform.omega.shrink(antimask),
                                  xform.frame, xform.reference, xform.origin)
            result.__xform_to_j2000_ = new_xform

        ssb = result.__ssb_
        if (ssb is not None and ssb is not result and
                                ssb.__xform_to_j2000_ is not None):
            xform = ssb.__xform_to_j2000_
            new_xform = Transform(xform.matrix.shrink(antimask),
                                  xform.omega.shrink(antimask),
                                  xform.frame, xform.reference, xform.origin)
            ssb.__xform_to_j2000_ = new_xform

        return result

    def unshrink(self, antimask):
        """Expand a shrunken version of this event to its original state.

        Antimask is None to leave the Event unchanged; otherwise, it must be a
        boolean array where True indicates values to be kept.
        """

        def unshrink1(arg):
            try:
                return arg.unshrink(antimask)
            except AttributeError:
                return arg

        if antimask is None: return self
        if antimask is True: return self
        if antimask is False: return self

        result = Event(self.time.unshrink(antimask),
                       self.state.unshrink(antimask),
                       self.origin, self.frame)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = unshrink1(self.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            if attr == '_Event__ssb_': continue
            result.__dict__[attr] = unshrink1(self.__dict__[attr])

        for (name, value) in self.subfields.iteritems():
            result.insert_subfield(name, unshrink1(value))

        if self.__ssb_ is self:
            result.__ssb_ = result
        elif self.__ssb_ is None:
            result.__ssb_ = None
        else:
            result.__ssb_ = self.__ssb_.unshrink(antimask)
            result.__ssb_._ssb_ = result.__ssb_

        result.__shape_ = None

        if result.__xform_to_j2000_ is not None:
            xform = result.__xform_to_j2000_
            new_xform = Transform(xform.matrix.unshrink(antimask),
                                  xform.omega.unshrink(antimask),
                                  xform.frame, xform.reference, xform.origin)
            result.__xform_to_j2000_ = new_xform

        ssb = result.__ssb_
        if (ssb is not None and ssb is not result and
                                ssb.__xform_to_j2000_ is not None):
            xform = ssb.__xform_to_j2000_
            new_xform = Transform(xform.matrix.unshrink(antimask),
                                  xform.omega.unshrink(antimask),
                                  xform.frame, xform.reference, xform.origin)
            ssb.__xform_to_j2000_ = new_xform

        return result

    ############################################################################
    # Event transformations
    ############################################################################

    def wrt_ssb(self, derivs=True, quick=False):
        """This event relative to SSB coordinates in the J2000 frame.

        This value is cached inside of the object so it can be quickly accessed
        again at a later time.

        Input:
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """


        if self.__ssb_ is None:
            if (self.__origin_ == Event.PATH_CLASS().SSB) and \
               (self.__frame_ == Frame.J2000):
                    self.__ssb_ = self
                    self.__ssb_.__ssb_ = self
                    self.__xform_to_j2000_ = Transform.identity(Frame.J2000)
            else:
                (self.__ssb_,
                 self.__xform_to_j2000_) = self.wrt(Event.PATH_CLASS().SSB,
                                                    Frame.J2000,
                                                    derivs=derivs, quick=quick,
                                                    include_xform=True)
                self.__ssb_.__ssb_ = self.__ssb_

        if self.__xform_to_j2000_ is None:
            frame = Frame.J2000.wrt(self.__frame_)
            self.__xform_to_j2000_ = frame.transform_at_time(self.__time_,
                                                             quick=quick)

        if derivs:
            return self.__ssb_
        else:
            return self.__ssb_.without_derivs()

    def from_ssb(self, path, frame, derivs=True, quick=False):
        """This SSB/J2000-relative event to a new path and frame.

        Input:
            path        the Path or path ID identifying the new origin;
                        None to leave the origin unchanged.
            frame       the Frame or frame ID of the new coordinate frame; None
                        to leave the frame unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        assert self.__frame_ == Frame.J2000
        assert self.__origin_ == Event.PATH_CLASS().SSB

        event = self.wrt(path, frame, derivs=True, quick=quick)
        event.__ssb_ = self
        event.__ssb_._ssb_ = self

        if derivs:
            return event
        else:
            return event.without_derivs()

    def wrt(self, path=None, frame=None, derivs=True, quick=False,
                  include_xform=False):
        """This event relative to a new path and/or a new coordinate frame.

        Input:
            path        the Path or path ID identifying the new origin;
                        None to leave the origin unchanged.
            frame       the Frame or frame ID of the new coordinate frame; None
                        to leave the frame unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            include_xform
                        if True, the transform is returned in a tuple along with
                        the new event.

        The new derivatives will multiply the originals, yielding derivatives
        with respect to the same event as those of this event.
        """

        # Interpret inputs
        if path is None: path = self.__origin_
        if frame is None: frame = self.__frame_

        path = Event.PATH_CLASS().as_path(path)
        frame = Frame.as_frame(frame)

        # Point to the working copy of this Event object
        event = self

        # If the path is shifting...
        xform1 = Transform.identity(event.__frame_)
        if event.__origin_.waypoint != path.waypoint:

            # ...and the current frame is rotating...
            old_frame = event.__frame_
            if old_frame.origin is not None:

                # ...then rotate to J2000
                (event, xform1) = event.wrt_frame(Frame.J2000,
                                                  derivs=derivs, quick=quick,
                                                  include_xform=True)

        # If the frame is changing...
        if event.__frame_.wayframe != frame.wayframe:

            # ...and the new frame is rotating...
            if frame.origin is not None:

                # ...then shift to the origin of the new frame
                event = event.wrt_path(frame.origin, derivs=derivs, quick=quick)

        # Now it is safe to rotate to the new frame
        (event, xform) = event.wrt_frame(frame, derivs=derivs, quick=quick,
                                         include_xform=True)

        # Now it is safe to shift to the new event
        result = event.wrt_path(path, derivs=derivs, quick=quick)

        if include_xform:
            return (result, xform.rotate_transform(xform1))
        else:
            return result

    def wrt_path(self, path, derivs=True, quick=False):
        """This event defined relative to a different origin path.

        The frame is unchanged.

        Input:
            path        the Path object to be used as the new origin. If the
                        value is None, the event is returned unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        if path is None: path = self.__origin_

        path = Event.PATH_CLASS().as_path(path)
        if self.__origin_.waypoint == path.waypoint:
            if derivs:
                return self
            else:
                return self.without_derivs()

        # Make sure frames match; make recursive calls to wrt() if needed
        event = self
        if self.frame.wayframe != path.frame.wayframe:
            event = event.wrt(path, path.frame, derivs=derivs, quick=quick)

        new_path = event.__origin_.wrt(path, path.frame)
        result = new_path.add_to_event(event, derivs=derivs, quick=quick)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = event.__dict__[attr]

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = event.__dict__[attr]

        if derivs:
            return result
        else:
            return result.without_derivs()

    def wrt_frame(self, frame, derivs=True, quick=False, include_xform=False):
        """This event defined relative to a different frame.

        The path is unchanged.

        Input:
            frame       the Frame object to be used as the new reference. If the
                        value is None, the event is returned unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            include_xform
                        if True, the transform is returned in a tuple along with
                        the new event.
        """

        if frame is None:
            frame = self.__frame_
        else:
            frame = Frame.as_frame(frame)

        new_frame = frame.wrt(self.__frame_)

        if self.__frame_.wayframe == frame.wayframe:
            if derivs:
                result = self
            else:
                result = self.without_derivs()

            if include_xform:
                return (result,
                        new_frame.transform_at_time(self.__time_,
                                                    quick=quick))
                        # Transform from event frame to new_frame
            else:
                return result

        return self.rotate_by_frame(new_frame, derivs=derivs, quick=quick,
                                               include_xform=include_xform)

    def rotate_by_frame(self, frame, derivs=True, quick=False,
                              include_xform=False):
        """This event rotated forward into a new frame.

        The origin is unchanged. Subfields are also rotated into the new frame.

        Input:
            frame       a Frame into which to transform the coordinates. Its
                        reference frame must be the current frame of the event.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            include_xform
                        if True, the transform is returned in a tuple along with
                        the new event.
        """

        def xform_rotate(arg):
            try:
                return xform.rotate(arg, derivs=True)
            except:
                return arg

        if derivs:
            event = self
        else:
            event = self.without_derivs()

        frame = Frame.as_frame(frame)
        xform = frame.transform_at_time(event.__time_, quick=quick)
                    # xform rotates from event frame to new frame

        state = xform.rotate(event.__state_, derivs=True)

        result = Event(event.__time_, state, event.__origin_, frame.wayframe)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = xform_rotate(event.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = event.__dict__[attr]

        result.__xform_to_j2000_ = None

        for (name, value) in event.__subfields_.iteritems():
            result.insert_subfield(name, xform_rotate(value))

        if include_xform:
            return (result, xform)
        else:
            return result

    def unrotate_by_frame(self, frame, derivs=True, quick=False):
        """This Event unrotated back into the given frame.

        The origin is unchanged. Subfields are also urotated.

        Input:
            frame       a Frame object to to inverse-transform the coordinates.
                        Its target frame must be the current frame of the event.
                        The returned event will use the reference frame instead.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them. Time derivatives are always
                        retained.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        def xform_unrotate(arg):
            try:
                return xform.unrotate(arg, derivs=True)
            except:
                return arg

        if derivs:
            event = self
        else:
            event = self.without_derivs()

        frame = Frame.as_frame(frame)
        xform = frame.transform_at_time(event.__time_, quick=quick)

        state = xform.unrotate(event.__state_, derivs=True)

        result = Event(event.__time_, state, event.__origin_, frame.reference)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = xform_unrotate(event.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            result.__dict__[attr] = event.__dict__[attr]

        result.__xform_to_j2000_ = None

        for (name, value) in event.__subfields_.iteritems():
            result.insert_subfield(name, xform_unrotate(value))

        return result

    def collapse_time(self, threshold=None):
        """If the time span is small, return a similar Event having fixed time.

        If the difference between the earliest time and the latest time is
        smaller than a specified threshold, a new Event object is returned in
        which the time is replaced by a Scalar equal to the midtime.

        Otherwise, the object is returned unchanged.

        Input:
            threshold   the allowed difference in seconds between the earliest
                        latest times. None to use the value specifed by the
                        EVENT_CONFIG.
        """

        def without_derivs(arg):
            if arg is None: return arg
            return arg.without_derivs()

        if self.__time_.shape == (): return self
        if self.__time_.derivs: return self

        if threshold is None:
            threshold = EVENT_CONFIG.collapse_threshold

        tmin = self.__time_.min()
        tmax = self.__time_.max()
        span = tmax - tmin

        collapsed_mask = (span == Scalar.MASKED)

        if span > threshold: return self

        if LOGGING.event_time_collapse:
            print LOGGING.prefix, "Event.collapse_time()",
            print tmin, tmax - tmin

        midtime = Scalar((tmin + tmax)/2., collapsed_mask, self.__time_.units)

        result = self.clone()
        result.__time_ = midtime

        if result.__ssb_ is not None and result.__ssb_ is not result:
            result.__ssb_.__time_ = midtime

        result.__shape_ = None

        return result

    ############################################################################
    # Event subtraction
    ############################################################################

    def sub(self, reference, quick={}):
        """The result of subtracting the reference event from this event.

        Used mainly for debugging. Note that the reference event could occur at
        a different time. Subtracted events can only be used to examine
        differences in properties; other methods will fail.

        Vectors in the returned object are in the frame of the reference event;
        times are relative. Photon events are unchanged except for the
        coordinate transform.

        The returned object has additional attributes 'event' and 'reference',
        which point to the source events
        """

        def ref_unrotate(arg):
            try:
                return reference.xform_to_j2000.unrotate(arg)
            except:
                return arg

        event_ssb = self.wrt_ssb(derivs=True, quick=quick)
        reference_ssb = reference.wrt_ssb(derivs=True, quick=quick)

        time = self.time - reference.time
        state = ref_unrotate(event_ssb.state - reference_ssb.state)
        diff = Event(time, state, reference.origin, reference.frame)

        diff.__ssb_ = self.__ssb_

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            diff.__dict__[attr] = ref_unrotate(event_ssb.__dict__[attr])

        for attr in Event.MORE_ATTRIBUTES:
            if attr == '_Event__ssb_': continue
            diff.__dict__[attr] = ref_unrotate(event_ssb.__dict__[attr])

        for (key,subfield) in event_ssb.__subfields_.iteritems():
            try:
                subfield = ref_unrotate(subfield)
            except:
                pass

            diff.insert_subfield(key, subfield)

        diff.event = self
        diff.reference = reference

        return diff

    ############################################################################
    # Aberration procedures
    ############################################################################

    def apparent_ray_ssb(self, ray_ssb, derivs=False, quick={}):
        """Apparent direction of a photon in the SSB/J2000 frame. Not cached.

        Input:
            ray_ssb     the true direction of a light ray in the SSB/J2000
                        system (not reversed!).
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # This procedure is equivalent to a vector subtraction of the velocity
        # of the observer from the ray, given that the ray has length C. However
        # the length of the ray is adjusted to be accurate to higher order in
        # (v/c)

        ray_ssb = Vector3.as_vector3(ray_ssb).as_readonly()

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        vel_ssb = wrt_ssb.vel + wrt_ssb.vflat

        if not derivs:
            ray_ssb = ray_ssb.without_derivs()
            vel_ssb = vel_ssb.without_derivs()

        # Below, factor = 1 is good to first order, matching the accuracy of the
        # SPICE toolkit. The expansion in beta below was determined empirically
        # to match the exact expression for the aberration, which is:
        #   tan(alpha'/2) = sqrt((1+beta)/(1-beta)) tan(alpha/2)
        #
        # alpha is the actual angle between the velocity vector and the photon's
        # direction of motion (NOT reversed).
        #
        # alpha' is the apparent angle.

        beta = C_INVERSE * vel_ssb.norm()
        ray_ssb_norm = ray_ssb.norm()
        cos_angle = C_INVERSE * vel_ssb.dot(ray_ssb) / (ray_ssb_norm * beta)
        factor = 1. - beta * (cos_angle - beta) * (0.5 + 0.375 * beta**2)

        return ray_ssb - (factor * C_INVERSE) * ray_ssb_norm * vel_ssb

    def actual_ray_ssb(self, ray_ap_ssb, derivs=False, quick={}):
        """Actual direction of a photon in the SSB/J2000 frame. Not cached.

        Input:
            ray_ap_ssb  the apparent direction of a light ray in the SSB/J2000
                        system.
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # This procedure is equivalent to a vector subtraction of the velocity
        # of the observer from the ray, given that the ray has length C.

        ray_ap_ssb = Vector3.as_vector3(ray_ap_ssb).as_readonly()

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        vel_ssb = wrt_ssb.vel + wrt_ssb.vflat

        if not derivs:
            ray_ap_ssb = ray_ap_ssb.without_derivs()
            vel_ssb = vel_ssb.without_derivs()

        # Invert the function above
        beta_ssb = C_INVERSE * vel_ssb
        beta = beta_ssb.norm()
        vel_inv = C_INVERSE / beta
        bb = beta * (0.5 + 0.375 * beta**2)
        f1 = 1. + bb * beta

        # Iterate solution
        ITERS = 4
        ray_ssb = ray_ap_ssb
        for iter in range(ITERS):
            ray_ssb_norm = ray_ssb.norm()
            cos_angle = vel_inv * vel_ssb.dot(ray_ssb) / ray_ssb_norm
            factor = f1 - bb * cos_angle
            ray_ssb = ray_ap_ssb + factor * ray_ssb_norm * beta_ssb

        return ray_ssb

    def apparent_arr(self, derivs=False, quick={}):
        """Apparent direction of an arriving ray in the event frame. Cached.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if not ABERRATION.old and self.__arr_ap_ is not None:
            if derivs:
                return self.__arr_ap_
            else:
                return self.__arr_ap_.without_derivs()

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        arr_ap_ssb = self.apparent_ray_ssb(wrt_ssb.arr, derivs, quick=quick)
        wrt_ssb.__arr_ap_ = arr_ap_ssb

        # Convert to this event's frame
        if self.__frame_ != Frame.J2000:
            arr_ap = self.__xform_to_j2000_.unrotate(arr_ap_ssb, derivs=True)
        else:
            arr_ap = arr_ap_ssb

        # Cache the result
        if not ABERRATION.old:
            self.__arr_ap_ = arr_ap

        if derivs:
            return arr_ap
        else:
            return arr_ap.without_derivs()

    def actual_arr(self, derivs=False, quick={}):
        """Actual direction of an arriving ray in the event frame. Cached

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if not ABERRATION.old and self.__arr_ is not None:
            if derivs:
                return self.__arr_
            else:
                return self.__arr_.without_derivs()

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        arr_ssb = self.actual_ray_ssb(wrt_ssb.arr_ap, derivs, quick=quick)
        wrt_ssb.__arr_ = arr_ssb

        # Convert to this event's frame
        if self.__frame_ != Frame.J2000:
            arr = self.__xform_to_j2000_.unrotate(arr_ssb, derivs=True)
        else:
            arr = arr_ssb

        # Cache the result
        if not ABERRATION.old:
            self.__arr_ = arr

        if derivs:
            return arr
        else:
            return arr.without_derivs()

    def apparent_dep(self, derivs=False, quick={}):
        """Apparent direction of a departing ray in the event frame. Cached.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if not ABERRATION.old and self.__dep_ap_ is not None:
            if derivs:
                return self.__dep_ap_
            else:
                return self.__dep_ap_.without_derivs()

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        dep_ap_ssb = self.apparent_ray_ssb(wrt_ssb.dep, derivs, quick=quick)
        wrt_ssb.__dep_ap_ = dep_ap_ssb

        # Convert to this event's frame
        if self.__frame_ != Frame.J2000:
            dep_ap = self.__xform_to_j2000_.unrotate(dep_ap_ssb, derivs=True)
        else:
            dep_ap = dep_ap_ssb

        # Cache the result
        if not ABERRATION.old:
            self.__dep_ap_ = dep_ap

        if derivs:
            return dep_ap
        else:
            return dep_ap.without_derivs()

    def actual_dep(self, derivs=False, quick={}):
        """Actual direction of a departing ray in the event frame. Cached.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if not ABERRATION.old and self.__dep_ is not None:
            if derivs:
                return self.__dep_
            else:
                return self.__dep_.without_derivs()

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs, quick=quick)
        dep_ssb = self.actual_ray_ssb(wrt_ssb.dep_ap, derivs, quick=quick)
        wrt_ssb.__dep_ = dep_ssb

        # Convert to this event's frame
        if self.__frame_ != Frame.J2000:
            dep = self.__xform_to_j2000_.unrotate(dep_ssb, derivs=True)
        else:
            dep = dep_ssb

        # Cache the result
        if not ABERRATION.old:
            self.__dep_ = dep

        if derivs:
            return dep
        else:
            return dep.without_derivs()

    def incidence_angle(self, apparent=False, derivs=False, quick={}):
        """The incidence angle.

        The incidence angle is measured between the surface normal and the
        reversed direction of the arriving photon.

        Input:
            apparent    True to account for the aberration in the Event frame.
            derivs      True to include the derivatives of the light ray in the
                        returned angle; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        ignore = self.wrt_ssb(derivs=True, quick=quick)

        if self.arr is None:
            raise ValueError('Undefined arrival vector in ' + str(self))

        if self.perp is None:
            raise ValueError('Undefined perpendicular vector in ' + str(self))

        if apparent:
            arr = self.arr_ap
        else:
            arr = self.arr

        return np.pi - self.perp.sep(arr, derivs)

    def emission_angle(self, apparent=False, derivs=False, quick={}):
        """The emission angle.

        The emission angle is measured between the surface normal and the
        direction of the departing photon.

        Input:
            apparent    True to account for the aberration in the Event frame.
            derivs      True to include any derivatives of the light ray in the
                        returned angle; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        ignore = self.wrt_ssb(derivs=True, quick=quick)

        if self.dep is None:
            raise ValueError('Undefined departure vector in ' + str(self))

        if self.perp is None:
            raise ValueError('Undefined perpendicular vector in ' + str(self))

        if apparent:
            dep = self.dep_ap
        else:
            dep = self.dep

        return self.perp.sep(dep, derivs)

    def phase_angle(self, apparent=False, derivs=False, quick={}):
        """The phase angle.

        The phase angle is measured between the apparent direction of the
        arriving photon and the reversed direction of the departing photon.

        Input:
            apparent    True to account for the aberration in the Event frame.
            derivs      True to include any derivatives of the light ray in the
                        returned angle; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        ignore = self.wrt_ssb(derivs=True, quick=quick)

        if self.arr is None:
            raise ValueError('Undefined arrival vector in ' + str(self))

        if self.dep is None:
            raise ValueError('Undefined departure vector in ' + str(self))

        if apparent:
            dep = self.dep_ap
            arr = self.arr_ap
        else:
            dep = self.dep
            arr = self.arr

        return np.pi - dep.sep(arr, derivs)

    def ra_and_dec(self, apparent=False, derivs=False,
                         subfield='arr', quick={}, frame='J2000'):
        """The right ascension and declination as a tuple of two Scalars.

        Input:
            apparent    True to include stellar aberration, thereby returning
                        the apparent direction of the photon relative to the
                        background stars; False to return the purely geometric
                        values, neglecting the motion of the observer.
            derivs      True to include any derivatives of the light ray in the
                        returned quantities; False to exclude them.
            subfield    The subfield to use for the calculation, either "arr"
                        or "dep". Note that an arriving direction is reversed.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
            frame       coordinate frame for RA and dec. Default is J2000. Use
                        None to use the frame of this event.
        """

        # Validate the inputs
        assert subfield in {'arr', 'dep'}

        # Identify the frame
        if frame == 'J2000' or frame == Frame.J2000:
            event = self.wrt_ssb(derivs=True, quick=quick)
        elif frame is None:
            event = self
        else:
            event = self.wrt_frame(frame, derivs=derivs, quick=quick)

        # Calculate the ray in J2000
        if not apparent:
            if subfield == 'arr':
                ray = event.neg_arr
            else:
                ray = event.dep
        elif ABERRATION.old:
            if subfield == 'arr':
                ray = -event.apparent_arr(derivs=derivs, quick=quick)
            else:
                ray = event.apparent_dep(derivs=derivs, quick=quick)
        else:
            if subfield == 'arr':
                ray = event.neg_arr_ap
            else:
                ray = event.dep_ap

        if ray is None:
            raise ValueError('Undefined light ray vector in ' + str(self))

        # Convert to RA and dec
        return ray.to_ra_dec_length(recursive=derivs)[:2]

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Event(unittest.TestCase):

    def setUp(self):
        import oops.body
        oops.body.Body.reset_registry()
        oops.body.define_solar_system('1990-01-01', '2010-01-01')

    def tearDown(self):
        pass

    def runTest(self):
        import cspyce

        # This is the exact formula for stellar aberration
        #   beta = v/c
        #   angle is measured from the direction of motion to the actual (not
        #       time-reversed) direction of the incoming ray.
        def aberrate(angle, beta):
            tan_half_angle_prime = np.sqrt((1.+beta) /
                                           (1.-beta)) * np.tan(angle/2.)
            return 2. * np.arctan(tan_half_angle_prime)

        def unaberrate(angle_prime, beta):
            tan_half_angle = np.sqrt((1.+beta) /
                                     (1.-beta)) * np.tan(angle_prime/2.)
            return 2. * np.arctan(tan_half_angle)

        # Test against the approximation sin(delta) = beta * sin(angle)
        # where angle_prime = angle + delta
        BETA = 0.001
        angles = np.arange(181.) * RPD
        exact_prime = aberrate(angles, BETA)
        delta = exact_prime - angles
        for k in range(181):
#             print k, np.sin(delta[k]), BETA * np.sin(angles[k]),
#             print np.sin(delta[k]) - BETA * np.sin(angles[k])
            self.assertTrue(abs(np.sin(delta[k]) - BETA * np.sin(angles[k])) <
                            1.e-6)

        ########################################################################
        # Test aberration magnitudes and directions to first order
        ########################################################################

        BETA = 0.001
        DEL = 3.e-9
        SPEED = BETA * C        # largest speed we care about is 300 km/s
        HALFPI = np.pi/2

        # Incoming aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev.arr = -Vector3.ZAXIS
        self.assertEqual(Vector3.ZAXIS.sep(ev.neg_arr_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr = -Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.neg_arr_ap) - (HALFPI-BETA)) < DEL)

        # Outgoing aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.dep_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.dep_ap) - (HALFPI+BETA)) < DEL)

        # Incoming aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr_ap = -Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.neg_arr_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr_ap = -Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.neg_arr) - (HALFPI+BETA)) < DEL)

        # Outgoing aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.dep_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep_ap = Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.dep) - (HALFPI-BETA)) < DEL)

        ########################################################################
        # Test compatibility with SPICE toolkit and with the exact calculation
        ########################################################################

        angles = np.arange(181.)
        cspyce_arr_ap = []
        cspyce_dep_ap = []
        for angle in angles:
            vobs = np.array([SPEED, 0., 0.])

            # Note the sign change on pobj, because we consider the photon's
            # direction, not the direction to the target
            pobj = np.array([-np.cos(angle * RPD), -np.sin(angle * RPD), 0.])
            appobj = cspyce.stelab(pobj, vobs)
            cspyce_arr_ap.append(np.arctan2(-appobj[1], -appobj[0]))

            pobj = np.array([np.cos(angle * RPD), np.sin(angle * RPD), 0.])
            appobj = cspyce.stlabx(pobj, vobs)
            cspyce_dep_ap.append(np.arctan2(appobj[1], appobj[0]))

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ray = Vector3.from_scalars(np.cos(angles * RPD),
                                   np.sin(angles * RPD), 0.)
        ev.arr = ray
        ev.dep = ray

        exact_arr_ap = aberrate(angles * RPD, BETA)
        exact_dep_ap = aberrate(angles * RPD, BETA)

        for k in range(181):
            arr_ap = np.arctan2(ev.arr_ap[k].vals[1], ev.arr_ap[k].vals[0])
            self.assertTrue(abs(cspyce_arr_ap[k] - exact_arr_ap[k]) < 1.e-6)
            self.assertTrue(abs(arr_ap - exact_arr_ap[k]) < 1.e-15)

        for k in range(181):
            dep_ap = np.arctan2(ev.dep_ap[k].vals[1], ev.dep_ap[k].vals[0])
            self.assertTrue(abs(cspyce_dep_ap[k] - exact_dep_ap[k]) < 1.e-6)
            self.assertTrue(abs(dep_ap - exact_dep_ap[k]) < 1.e-15)

        ########################################################################
        # Test aberration inversions
        ########################################################################

        COUNT = 2000
        ev1 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev1.arr_ap = Vector3.from_scalars(np.random.randn(COUNT),
                                          np.random.randn(COUNT),
                                          np.random.randn(COUNT))
        ev1.dep_ap = Vector3.from_scalars(np.random.randn(COUNT),
                                          np.random.randn(COUNT),
                                          np.random.randn(COUNT))

        ev2 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev2.arr = ev1.arr
        ev2.dep = ev1.dep

        self.assertTrue((ev2.arr_ap.unit() -
                         ev1.arr_ap.unit()).norm().max() < 1.e-15)

        ev1 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev1.arr = Vector3.from_scalars(np.random.randn(COUNT),
                                       np.random.randn(COUNT),
                                       np.random.randn(COUNT))
        ev1.dep = Vector3.from_scalars(np.random.randn(COUNT),
                                       np.random.randn(COUNT),
                                       np.random.randn(COUNT))

        ev2 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev2.arr_ap = ev1.arr_ap
        ev2.dep_ap = ev1.dep_ap

        self.assertTrue((ev2.arr_ap.unit() -
                         ev1.arr_ap.unit()).norm().max() < 1.e-15)

        ########################################################################
        # Subfield checks
        ########################################################################

        for (origin, frame) in [('SSB', 'J2000'),
                                ('EARTH', 'IAU_EARTH'),
                                ('PLUTO', 'IAU_EARTH')]:

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ########################
            # Define arr
            ########################

            ev.arr = (1,2,3)
            self.assertEqual(ev._Event__arr_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            self.assertIsNone(ev._Event__neg_arr_)
            self.assertEqual(ev.neg_arr, Vector3((-1.,-2.,-3.)))
            self.assertIs(ev.neg_arr, ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            # Let arr_ap and ssb be filled in
            ignore = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define arr_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.arr_ap = (1,2,3)
            self.assertEqual(ev._Event__arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            self.assertEqual(ev.neg_arr_ap, Vector3((-1.,-2.,-3.)))
            self.assertEqual(ev._Event__arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__ssb_)

            # Let arr and ssb be filled in
            ignore = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertIs(ev.arr_j2000, ev.arr)
                self.assertIs(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define arr_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.arr_j2000 = (1,2,3)
            self.assertIsNotNone(ev._Event__ssb_)
            self.assertEqual(ev.ssb._Event__arr_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev.ssb._Event__arr_ap_)
            self.assertIsNone(ev.ssb._Event__neg_arr_)
            self.assertIsNone(ev.ssb._Event__neg_arr_ap_)

            self.assertIsNotNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNotNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            # Let arr_ap and ssb be filled in
            ignore = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define arr_ap_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.arr_ap_j2000 = (1,2,3)
            self.assertIsNotNone(ev._Event__ssb_)
            self.assertEqual(ev.ssb._Event__arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev.ssb._Event__arr_)
            self.assertIsNone(ev.ssb._Event__neg_arr_)
            self.assertIsNone(ev.ssb._Event__neg_arr_ap_)

            self.assertIsNotNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNotNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)

            # Let arr and ssb be filled in
            ignore = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define neg_arr
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.neg_arr = (-1,-2,-3)
            self.assertEqual(ev._Event__arr_, Vector3((1.,2.,3.)))
            self.assertEqual(ev._Event__neg_arr_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            # Let arr_ap and ssb be filled in
            ignore = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define neg_arr_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.neg_arr_ap = (-1,-2,-3)
            self.assertEqual(ev._Event__arr_ap_, Vector3((1.,2.,3.)))
            self.assertEqual(ev._Event__neg_arr_ap_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__ssb_)

            # Let arr and ssb be filled in
            ignore = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define neg_arr_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.neg_arr_j2000 = (-1,-2,-3)
            self.assertIsNotNone(ev._Event__ssb_)
            self.assertEqual(ev.ssb._Event__arr_, Vector3((1.,2.,3.)))
            self.assertEqual(ev.ssb._Event__neg_arr_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev.ssb._Event__arr_ap_)
            self.assertIsNone(ev.ssb._Event__neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            # Let arr_ap and ssb be filled in
            ignore = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define neg_arr_ap_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__arr_)
            self.assertIsNone(ev._Event__arr_ap_)
            self.assertIsNone(ev._Event__neg_arr_)
            self.assertIsNone(ev._Event__neg_arr_ap_)
            self.assertIsNone(ev._Event__ssb_)

            ev.neg_arr_ap_j2000 = (-1,-2,-3)
            self.assertIsNotNone(ev._Event__ssb_)
            self.assertEqual(ev.ssb._Event__arr_ap_, Vector3((1.,2.,3.)))
            self.assertEqual(ev.ssb._Event__neg_arr_ap_, Vector3((-1.,-2.,-3.)))
            self.assertIsNotNone(ev.ssb._Event__arr_ap_)
            self.assertIsNotNone(ev.ssb._Event__neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            # Let arr and ssb be filled in
            ignore = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.arr)
                self.assertIsNotNone(ev._Event__ssb_.arr_ap)

            ########################
            # Define dep
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__dep_)
            self.assertIsNone(ev._Event__dep_ap_)

            ev.dep = (-1,2,-3)
            self.assertEqual(ev._Event__dep_, Vector3((-1.,2.,-3.)))
            self.assertIsNone(ev._Event__dep_ap_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.dep_ap = (-1,2,-3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__dep_ap_)
            self.assertIsNone(ev._Event__ssb_)

            # Fill in dep_ap and ssb
            ignore = ev.dep_ap
            self.assertTrue((ev.dep_ap - ev.dep).norm() < 5*BETA)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.dep_j2000, ev.dep)
                self.assertEqual(ev.dep_ap_j2000, ev.dep_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.dep)
                self.assertIsNotNone(ev._Event__ssb_.dep_ap)

            ########################
            # Define dep_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._Event__dep_)
            self.assertIsNone(ev._Event__dep_ap_)

            ev.dep_ap = (-1,2,-3)
            self.assertEqual(ev._Event__dep_ap_, Vector3((-1.,2.,-3.)))
            self.assertIsNone(ev._Event__dep_)
            self.assertIsNone(ev._Event__ssb_)

            try:
                ev.dep_ap = (-1,2,-3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._Event__dep_)
            self.assertIsNone(ev._Event__ssb_)

            # Fill in dep and ssb
            ignore = ev.dep
            self.assertTrue((ev.dep_ap - ev.dep).norm() < 5*BETA)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._Event__ssb_)
                self.assertEqual(ev.dep_j2000, ev.dep)
                self.assertEqual(ev.dep_ap_j2000, ev.dep_ap)
            else:
                self.assertIsNotNone(ev._Event__ssb_)
                self.assertIsNotNone(ev._Event__ssb_.dep)
                self.assertIsNotNone(ev._Event__ssb_.dep_ap)

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
