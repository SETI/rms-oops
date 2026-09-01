##########################################################################################
# oops/event.py
##########################################################################################

import numpy as np

from polymath          import Qube, Scalar, Vector3
from oops.config       import EVENT_CONFIG, LOGGING
from oops.constants    import C_INVERSE
from oops.frame.frame_ import Frame
from oops.transform    import Transform


class Event(object):
    """An Event is defined by a time, position and velocity.

    Each property below is also accessible through a property of the same name without
    the surrounding underscores.

    Events are intended to be immutable. The exception is that the optional properties
    can be set exactly once after the constructor is called. You can define the photon
    directions with either apparent or actual values, but not both; whichever you define,
    the other is generated as needed.

    The properties of an Event need not have the same shape, but they must all be
    broadcastable to the same shape.

    Properties:
        * _time_ (Scalar): Event time of arbitrary shape, in seconds TDB relative to noon
          TDB on January 1, 2000, consistent with the time system used by the SPICE
          toolkit.
        * _state_ (Vector3): Position of the event, of arbitrary shape, in km relative to
          `_origin_`. Velocities are carried in km/s as the "t" derivative of the
          position.
        * _origin_ (Path): The path defining the location relative to which all positions
          and velocities are measured.
        * _frame_ (Frame): The frame defining the coordinate system in which the
          components of the positions and velocities are defined.
        * _subfields_ (dict): An arbitrary dictionary of objects providing further
          information about the properties of the event.
        * _arr_ (Vector3): The direction of a photon arriving at this event, defined in
          the frame of this event and not corrected for stellar aberration. Its length is
          arbitrary.
        * _arr_ap_ (Vector3): As `_arr_`, but the apparent direction of the arriving
          photon.
        * _arr_j2000_ (Vector3): As `_arr_`, but in J2000 coordinates.
        * _arr_ap_j2000_ (Vector3): As `_arr_ap_`, but in J2000 coordinates.
        * _neg_arr_ (Vector3): Negative of `_arr_`, because it is used so often.
        * _neg_arr_ap_ (Vector3): Negative of `_arr_ap_`.
        * _neg_arr_j2000_ (Vector3): Negative of `_arr_j2000_`.
        * _neg_arr_ap_j2000_ (Vector3): Negative of `_arr_ap_j2000_`.
        * _arr_lt_ (Scalar): The (negative) light travel time for the arriving photon from
          its origin.
        * _dep_ (Vector3): The direction of a photon departing from this event, defined in
          the frame of this event and not corrected for stellar aberration. Its length is
          arbitrary.
        * _dep_ap_ (Vector3): As `_dep_`, but the apparent direction of the departing
          photon.
        * _dep_j2000_ (Vector3): As `_dep_`, but in J2000 coordinates.
        * _dep_ap_j2000_ (Vector3): As `_dep_ap_`, but in J2000 coordinates.
        * _dep_lt_ (Scalar): The light travel time of a departing photon to its
          destination.
        * _perp_ (Vector3 or None): The direction of a normal vector if this event falls
          on a surface.
        * _vflat_ (Vector3): A velocity component within the surface, which can be used to
          describe winds across a planet or orbital motion within a ring plane.
        * _ssb_ (Event): This event referenced to SSB/J2000.
        * _xform_to_j2000_ (Transform): The transform that converts coordinates in this
          event to J2000.
    """

    # To avoid circular imports; filled in by oops/__init__.py
    _Path = None
    SSB = None

    # Property names, categorized
    ARR_VEC3_PROPERTIES = ['arr', 'arr_ap', 'arr_j2000', 'arr_ap_j2000', 'neg_arr',
                           'neg_arr_ap', 'neg_arr_j2000', 'neg_arr_ap_j2000']

    DEP_VEC3_PROPERTIES = ['dep', 'dep_ap', 'dep_j2000', 'dep_ap_j2000']

    SPECIAL_PROPERTIES = ARR_VEC3_PROPERTIES + DEP_VEC3_PROPERTIES
    SPECIAL_PROPERTIES += ['arr_lt', 'dep_lt', 'perp', 'vflat']

    @staticmethod
    def attr_name(prop_name):
        return '_' + prop_name + '_'

    def get_prop(self, prop_name):
        return Event.__dict__[prop_name].fget(self)

    def set_prop(self, prop_name, value):
        Event.__dict__[prop_name].fset(self, value)

    def __init__(self, time, state, origin, frame=None, **more):
        """Constructor for the Event class.

        Parameters:
            time (Scalar): Event times in seconds TDB.
            state (Vector3): Position vectors as a Vector3 object. The velocity should be
                included as the time-derivative. However, if specified as a tuple of two
                objects, the first is interpreted as the position and the second as the
                velocity.
            origin (Path or str): The path or path ID identifying the origin of this
                event.
            frame (Frame, optional): The frame or frame ID identifying the coordinate
                frame of this event. Default is the frame of the origin path.
            **more: An arbitrary set of properties and subfields that will also be
                accessible as attributes of the Event object. Properties have fixed names
                and purposes; subfields can be anything.
        """

        self._time_ = Scalar.as_scalar(time).as_readonly()

        if isinstance(state, (tuple,list)) and len(state) == 2:
            pos = Vector3.as_vector3(state[0])
            vel = Vector3.as_vector3(state[1])
            state = pos.with_deriv('t', vel, method='insert')
        else:
            state = Vector3.as_vector3(state)
            if 't' not in state.derivs:
                state = state.with_deriv('t', Vector3.ZERO, method='insert')

        self._state_ = state.as_readonly()
        self._pos_ = self._state_.without_deriv('t')
        self._origin_ = Event._Path.as_waypoint(origin)
        if frame is None:
            frame = self._origin_.frame
        self._frame_ = Frame.as_wayframe(frame)

        self._ssb_ = None
        self._xform_to_j2000_ = None
        self._shape_ = None
        self._mask_ = None
        self._antimask_ = None
        self._wod_ = None

        # Set default values for properties
        for prop_name in Event.SPECIAL_PROPERTIES:
            self.__dict__[Event.attr_name(prop_name)] = None

        # Fill in any given subfields or properties
        self._subfields_ = {}
        for (name, value) in more.items():
            self.insert_subfield(name, value)

    def __getstate__(self):
        """The minimum info necessary to preserve the entire state of the event.
        """

        more = {}           # dict of the defined photon properties and subfields

        # Save only the first defined arriving photon vector; the rest are derivable
        for prop in Event.ARR_VEC3_PROPERTIES:
            vec = getattr(self, Event.attr_name(prop))
            if vec is not None:
                more[prop] = vec
                break

        # Save only the first defined departing photon vector; the rest are derivable
        for prop in Event.DEP_VEC3_PROPERTIES:
            vec = getattr(self, Event.attr_name(prop))
            if vec is not None:
                more[prop] = vec
                break

        # Save additional properties if defined
        for prop in ('arr_lt', 'dep_lt', 'perp', 'vflat'):
            value = getattr(self, Event.attr_name(prop))
            if value is not None:
                more[prop] = value

        # Save the subfields
        for (key, value) in self.subfields.items():
            more[key] = value

        return (self._time_, self._state_, self._origin_, self._frame_, more)

    def __setstate__(self, state):
        self.__init__(*state[:-1], **state[-1])

    ######################################################################################
    # Read-only properties
    ######################################################################################

    @property
    def time(self):
        """Event times in seconds TDB."""

        return self._time_

    @property
    def state(self):
        """Position with velocity as time-derivative .d_dt."""
        return self._state_

    @property
    def pos(self):
        """Position without velocity as time-derivative."""
        return self._pos_

    @property
    def vel(self):
        """Event velocities in km/s, the time-derivative of the position."""

        if hasattr(self._state_, 'd_dt'):
            return self._state_.d_dt
        else:
            return Vector3.ZERO

    @property
    def origin(self):
        """The Path defining where positions and velocities are measured from."""

        return self._origin_

    @property
    def origin_id(self):
        """The ID of the origin Path."""

        return self._origin_.path_id

    @property
    def frame(self):
        """The Frame in which the position and velocity components are defined."""

        return self._frame_

    @property
    def frame_id(self):
        """The ID of the coordinate Frame."""

        return self._frame_.frame_id

    @property
    def subfields(self):
        """The dictionary of further information about this event."""

        return self._subfields_

    @property
    def shape(self):
        """The shape of this Event, broadcast across all of its properties."""

        if self._shape_ is None:
            self._shape_ = Qube.broadcasted_shape(self._time_, self._state_,
                                                  self._origin_, self._frame_, self._arr_,
                                                  self._arr_ap_, self._dep_,
                                                  self._dep_ap_)
        return self._shape_

    @property
    def size(self):
        """The number of elements in this Event."""

        return int(np.prod(self.shape))

    @property
    def mask(self):
        """The mask, True where this Event is undefined."""

        if self._mask_ is None:
            self._mask_ = Qube.or_(self._time_.mask, self._state_.mask, self.vel.mask)
            if self._dep_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._dep_.mask)
            if self._dep_ap_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._dep_ap_.mask)
            if self._dep_lt_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._dep_lt_.mask)
            if self._arr_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._arr_.mask)
            if self._arr_ap_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._arr_ap_.mask)
            if self._arr_lt_ is not None:
                self._mask_ = Qube.or_(self._mask_, self._arr_lt_.mask)

            self._antimask_ = None

        return self._mask_

    @property
    def antimask(self):
        """The antimask, True where this Event is defined."""

        if self._antimask_ is None:
            self._antimask_ = np.logical_not(self.mask)

        return self._antimask_

    @property
    def ssb(self):
        """This Event referenced to SSB/J2000, evaluated on first use."""

        if self._ssb_ is None:
            _ = self.wrt_ssb(derivs=True)

        return self._ssb_

    @property
    def xform_to_j2000(self):
        """Transform that rotates from event coordinates to J2000 coordinates.
        """

        if self._xform_to_j2000_ is None:
            if self._ssb_ is None:
                _ = self.wrt_ssb(derivs=True)
            else:
                self._xform_to_j2000_ = self.wrt(Event.SSB, Frame.J2000, derivs=True,
                                                 quick=None, include_xform=True)[1]

        return self._xform_to_j2000_

    @property
    def wod(self):
        """This Event without any derivatives, evaluated on first use."""

        if self._wod_ is None:
            self._wod_ = self.without_derivs()
            self._wod_._wod_ = self._wod_

        return self._wod_

    def empty_cache(self):
        """Remove cached properties; call every time an attribute is set."""

        self._wod_ = None
        self._mask_ = None
        self._antimask_ = None
        self._shape_ = None

        if self._ssb_:
            self._ssb_._wod_ = None
            self._ssb_._mask_ = None
            self._ssb_._antimask_ = None
            self._ssb_._shape_ = None

    def _refresh(self):
        """Remove all internal information; needed for Events that involve Fittable
        objects.
        """

        self._ssb_ = None
        self._xform_to_j2000_ = None
        self._shape_ = None
        self._mask_ = None
        self._antimask_ = None
        self._wod_ = None

    def has_arrivals(self):
        """True if arrival photons have been defined for this event."""

        return self._arr_ is not None or self._arr_ap_ is not None

    def has_departures(self):
        """True if departure photons have been defined for this event."""

        return self._dep_ is not None or self._dep_ap_ is not None

    ######################################################################################
    # Special properties: Photon arrival vectors
    #
    # These values are cached for repeated use.
    #
    # Upon setting any of these parameters, the immediate value is saved and at least one
    # of the attributes _arr_ap_ and _arr_ is filled in. All other attributes of arriving
    # photons are derived from one of these. Each of these can be derived from the other
    # using actual_arr() and apparent_arr().
    ######################################################################################

    @property
    def arr(self):
        """The direction of a photon arriving at this event, in its own frame."""

        if self._arr_ is None:
            if self._arr_ap_ is not None:
                _ = self.actual_arr(derivs=True)    # fill internal attribute

        return self._arr_   # returns None if still undefined

    @arr.setter
    def arr(self, value):
        if (self._arr_ is not None) or (self._arr_ap_ is not None):
            raise ValueError(f'arriving photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        arr = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, arr)

        self._arr_ = arr
        if (self._ssb_ is not None) and (self._ssb_._arr_ is None):
            ssb_arr = self.xform_to_j2000.rotate(self._arr_)
            self._ssb_._arr_ = ssb_arr.as_readonly()

        self.empty_cache()

    @property
    def arr_ap(self):
        """The apparent direction of a photon arriving at this event."""

        if self._arr_ap_ is None:
            if self._arr_ is not None:
                _ = self.apparent_arr(derivs=True)  # fill internal attribute

        return self._arr_ap_    # returns None if still undefined

    @arr_ap.setter
    def arr_ap(self, value):
        if (self._arr_ap_ is not None) or (self._arr_ is not None):
            raise ValueError(f'arriving photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        arr_ap = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, arr_ap)

        self._arr_ap_ = arr_ap
        if (self._ssb_ is not None) and (self._ssb_._arr_ap_ is None):
            ssb_arr_ap = self.xform_to_j2000.rotate(self._arr_ap_)
            self._ssb_._arr_ap_ = ssb_arr_ap.as_readonly()

        self.empty_cache()

    @property
    def arr_j2000(self):
        """The direction of an arriving photon, in J2000 coordinates."""

        return self.ssb.arr

    @arr_j2000.setter
    def arr_j2000(self, value):
        ssb_event = self.ssb
        if self is ssb_event:       # avoid recursion
            self.arr = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.arr = self.xform_to_j2000.unrotate(value)
            ssb_event._arr_ = value

        self.empty_cache()

    @property
    def arr_ap_j2000(self):
        """The apparent direction of an arriving photon, in J2000 coordinates."""

        return self.ssb.arr_ap

    @arr_ap_j2000.setter
    def arr_ap_j2000(self, value):
        ssb_event = self.ssb
        if self is ssb_event:       # avoid recursion
            self.arr_ap = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.arr_ap = self.xform_to_j2000.unrotate(value)
            ssb_event._arr_ap_ = value

        self.empty_cache()

    @property
    def arr_lt(self):
        """The light travel time of an arriving photon from its source, negative."""

        return self._arr_lt_        # returns None if still undefined

    @arr_lt.setter
    def arr_lt(self, value):
        if self._arr_lt_ is not None:
            raise ValueError(f'arriving photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        arr_lt = Scalar.as_scalar(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, arr_lt)

        self._arr_lt_ = arr_lt
        if (self._ssb_ is not None) and (self._ssb_._arr_lt_ is None):
            self._ssb_._arr_lt_ = self._arr_lt_

        self.empty_cache()

    ######################################################################################
    # Special properties: Photon arrival vectors, reversed
    #
    # These values are cached for repeated use.
    #
    # Upon setting any of these parameters, the immediate value is saved and at least one
    # of the attributes _arr_ap_ and _arr_ is filled in. All other attributes of arriving
    # photons are derived from one of these.
    ######################################################################################

    @property
    def neg_arr(self):
        """The negative of `arr`."""

        if self._neg_arr_ is None and self.arr is not None:
            self._neg_arr_ = -self.arr

        return self._neg_arr_

    @neg_arr.setter
    def neg_arr(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.arr = -value
        self._neg_arr_ = value

        self.empty_cache()

    @property
    def neg_arr_ap(self):
        """The negative of `arr_ap`."""

        if self._neg_arr_ap_ is None and self.arr_ap is not None:
            self._neg_arr_ap_ = -self.arr_ap

        return self._neg_arr_ap_

    @neg_arr_ap.setter
    def neg_arr_ap(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.arr_ap = -value
        self._neg_arr_ap_ = value

        self.empty_cache()

    @property
    def neg_arr_j2000(self):
        """The negative of `arr_j2000`."""

        return self.ssb.neg_arr

    @neg_arr_j2000.setter
    def neg_arr_j2000(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.ssb.arr = -value
        self.ssb._neg_arr_ = value

        if self.ssb is not self:        # avoid recursion
            self.arr = self.xform_to_j2000.unrotate(self.ssb._arr_)

        self.empty_cache()

    @property
    def neg_arr_ap_j2000(self):
        """The negative of `arr_ap_j2000`."""

        return self.ssb.neg_arr_ap

    @neg_arr_ap_j2000.setter
    def neg_arr_ap_j2000(self, value):
        value = Vector3.as_vector3(value).as_readonly()
        self.ssb.arr_ap = -value
        self.ssb._neg_arr_ap_ = value

        if self.ssb is not self:        # avoid recursion
            self.arr_ap = self.xform_to_j2000.unrotate(self.ssb._arr_ap_)

        self.empty_cache()

    ######################################################################################
    # Special properties: Photon departure vectors
    #
    # These values are cached for repeated use.
    #
    # Upon setting any of these parameters, the immediate value is saved and at least one
    # of the attributes _dep_ap_ and _dep_ is filled in. All other attributes of departing
    # photons are derived from one of these. Each of these can be derived from the other
    # using actual_dep() and apparent_dep().
    ######################################################################################

    @property
    def dep(self):
        """The direction of a photon departing from this event, in its own frame."""

        if self._dep_ is None:
            if self._dep_ap_ is not None:
                _ = self.actual_dep(derivs=True)    # fill internal attribute

        return self._dep_   # returns None if still undefined

    @dep.setter
    def dep(self, value):
        if (self._dep_ is not None) or (self._dep_ap_ is not None):
            raise ValueError(f'departing photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        dep = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, dep)

        self._dep_ = dep
        if (self._ssb_ is not None) and (self._ssb_._dep_ is None):
            ssb_dep = self.xform_to_j2000.rotate(self._dep_)
            self._ssb_._dep_ = ssb_dep.as_readonly()

        self.empty_cache()

    @property
    def dep_ap(self):
        """The apparent direction of a photon departing from this event."""

        if self._dep_ap_ is None:
            if self._dep_ is not None:
                _ = self.apparent_dep(derivs=True)  # fill internal attribute

        return self._dep_ap_

    @dep_ap.setter
    def dep_ap(self, value):
        if (self._dep_ap_ is not None) or (self._dep_ is not None):
            raise ValueError(f'departing photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        dep_ap = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, dep_ap)

        self._dep_ap_ = dep_ap

        if (self._ssb_ is not None) and (self._ssb_._dep_ap_ is None):
            ssb_dep_ap = self.xform_to_j2000.rotate(self._dep_ap_)
            self._ssb_._dep_ap_ = ssb_dep_ap.as_readonly()

        self.empty_cache()

    @property
    def dep_j2000(self):
        """The direction of a departing photon, in J2000 coordinates."""

        return self.ssb.dep

    @dep_j2000.setter
    def dep_j2000(self, value):
        ssb_event = self.ssb

        if self is ssb_event:       # avoid recursion
            self.dep = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.dep = self.xform_to_j2000.unrotate(value)
            ssb_event._dep_ = value

        self.empty_cache()

    @property
    def dep_ap_j2000(self):
        """The apparent direction of a departing photon, in J2000 coordinates."""

        return self.ssb.dep_ap

    @dep_ap_j2000.setter
    def dep_ap_j2000(self, value):
        ssb_event = self.ssb
        if self is ssb_event:       # avoid recursion
            self.dep_ap = value
        else:
            value = Vector3.as_vector3(value).as_readonly()
            self.dep_ap = self.xform_to_j2000.unrotate(value)
            ssb_event._dep_ap_ = value.as_readonly()

        self.empty_cache()

    @property
    def dep_lt(self):
        """The light travel time of a departing photon to its destination."""

        return self._dep_lt_

    @dep_lt.setter
    def dep_lt(self, value):
        if self._dep_lt_ is not None:
            raise ValueError(f'departing photons were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        dep_lt = Scalar.as_scalar(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, dep_lt)

        self._dep_lt_ = dep_lt

        if (self._ssb_ is not None) and (self._ssb_._dep_lt_ is None):
            self._ssb_._dep_lt_ = self._dep_lt_

        self.empty_cache()

    ######################################################################################
    # Special properties: Additional surface properties
    ######################################################################################

    @property
    def perp(self):
        """The normal vector where this event falls on a surface, None if undefined."""

        return self._perp_

    @perp.setter
    def perp(self, value):
        if self._perp_ is not None:
            raise ValueError(f'perpendiculars were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        perp = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, perp)

        self._perp_ = perp

        if (self._ssb_ is not None) and (self._ssb_._perp_ is None):
            ssb_perp = self.xform_to_j2000.rotate(self._perp_)
            self._ssb_._perp_ = ssb_perp.as_readonly()

        self.empty_cache()

    @property
    def vflat(self):
        """The velocity component within the surface, zero if it was never defined.

        The default is not saved, because doing so would count as defining the value and
        would block any later assignment.

        Returns:
            Vector3: The surface velocity, or `Vector3.ZERO` if none was assigned.
        """

        if self._vflat_ is None:
            return Vector3.ZERO

        return self._vflat_

    @vflat.setter
    def vflat(self, value):
        if self._vflat_ is not None:
            raise ValueError(f'surface velocities were already defined in {self}')

        # Raise a ValueError if the shape is incompatible
        vflat = Vector3.as_vector3(value).as_readonly()
        self._shape_ = Qube.broadcasted_shape(self.shape, vflat)

        self._vflat_ = vflat

        if (self._ssb_ is not None) and (self._ssb_._vflat_ is None):
            ssb_vflat = self.xform_to_j2000.rotate(self._vflat_)
            self._ssb_._vflat_ = ssb_vflat.as_readonly()

        self.empty_cache()

    ######################################################################################
    # Standard methods
    ######################################################################################

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
                     self._origin_.path_id, ', ',
                     self._frame_.frame_id]

        keys = list(self._subfields_.keys())
        keys.sort()
        for key in keys:
            str_list += ['; ', key]

        str_list += [')']
        return ''.join(str_list)

    ######################################################################################
    # Subfield and property methods
    ######################################################################################

    def insert_subfield(self, name, value):
        """Insert a given subfield into this Event."""

        if name in Event.SPECIAL_PROPERTIES:
            self.set_prop(name, value)

        else:
            self.__dict__[name] = value
            self._subfields_[name] = value

            if self._ssb_ is not None and self._ssb_ is not self:
                try:
                    value_j2000 = self.xform_to_j2000.rotate(value)
                except (ValueError, TypeError, KeyError):
                    value_j2000 = value

                self._ssb_.insert_subfield(name, value_j2000)

        self.empty_cache()

    def get_subfield(self, name):
        """The value of a given subfield or property."""

        if name in Event.SPECIAL_PROPERTIES:
            return self.get_prop(name)

        return self.subfields[name]

    ######################################################################################
    # Constructors for variant Event objects
    ######################################################################################

    def _apply_this_func(self, func, *args):
        """A new event in which the given function has been applied to every
        attribute.

        Parameters:
            func (callable): Function to apply to each Qube attribute of this Event.
            *args: Additional arguments to pass to `func` after the attribute value.

        Returns:
            Event: The new Event, with `func` applied to every attribute.
        """

        # Create the new event
        result = Event(func(self._time_, *args),
                       func(self._state_, *args),
                       self._origin_, self._frame_)

        # Apply to all the properties
        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            value = self.__dict__[attr]
            if isinstance(value, Qube):
                result.__dict__[attr] = func(value, *args)
            else:
                result.__dict__[attr] = value

        # Handle SSB attributes
        if self._ssb_ is None:
            result._ssb_ = None
        elif self._ssb_ == self:
            result._ssb_ = result
        else:
            result._ssb_ = self._ssb_._apply_this_func(func, *args)
            result._ssb_._ssb_ = result._ssb_
            result._xform_to_j2000_ = self.xform_to_j2000

        # Handle subfields
        for (name, value) in self._subfields_.items():
            if isinstance(value, Qube):
                result.insert_subfield(name, func(value, *args))
            else:
                result.insert_subfield(name, value)

        return result

    def copy(self, omit=()):
        """A shallow copy of the Event.

        Parameters:
            omit (list): Names of properties and subfields to omit. Use 'arr' to omit all
                arrival vectors and 'dep' to omit all departure vectors; other properties
                and subfields must be named explicitly.
        """

        def clone_attr(arg):
            return arg.clone(recursive=True)

        result = self._apply_this_func(clone_attr)

        if not isinstance(omit, (tuple,list)):
            omit = [omit]

        # Expand the list of omissions
        omissions = []
        for name in omit:
            if name == 'arr':
                omissions += Event.ARR_VEC3_PROPERTIES
            elif name == 'dep':
                omissions += Event.DEP_VEC3_PROPERTIES
            else:
                omissions.append(name)

        # Handle the omissions
        for name in omissions:

            # Wipe out a property
            if name in Event.SPECIAL_PROPERTIES:
                attr = Event.attr_name(name)
                result.__dict__[attr] = None
                if result._ssb_ is not None:
                    result._ssb_.__dict__[attr] = None

            # Otherwise assume it is a subfield
            else:
                try:
                    del result.subfields[name]
                except KeyError:
                    pass

                try:
                    del result.__dict__[name]
                except KeyError:
                    pass

                if result._ssb_:
                    try:
                        del result._ssb_.subfields[name]
                    except KeyError:
                        pass

                    try:
                        del result._ssb_.__dict__[name]
                    except KeyError:
                        pass

        return result

    def without_derivs(self):
        """A shallow copy of this Event without any derivatives except time. Unlike the
        .wod property, this version does not cache the result.
        """

        def remove_derivs(arg):
            return arg.without_derivs(preserve='t')

        return self._apply_this_func(remove_derivs)

    def as_all_masked(self, origin=None, frame=None, *, broadcast=None):
        """A shallow copy of this event, entirely masked.

        Parameters:
            origin (Path or str, optional): The origin or origin_id of the Event returned;
                if None, use the origin of this Event.
            frame (Frame or str, optional): The frame or frame_id of the Event returned;
                if None, use the frame of this Event.
            broadcast (tuple, optional): The new shape to broadcast the result into; None
                to leave the shape unchanged.
        """

        def fully_masked(arg):
            return arg.as_all_masked().broadcast_to(broadcast)

        if broadcast is None:
            broadcast = self.shape

        result = self._apply_this_func(fully_masked)
        result._mask_ = True
        result._antimask_ = False

        # Change the origin or frame if requested
        if origin:
            result._origin_ = origin
        if frame:
            result._frame_ = frame

        # Fill in _ssb_, also masked
        if (result._origin_ == Event.SSB and result._frame_ == Frame.J2000):
            result._ssb_ = result
        else:
            result._ssb_ = result.as_all_masked(Event.SSB, Frame.J2000)
            result._ssb_._xform_to_j2000_ = Transform.IDENTITY

        if result._xform_to_j2000_ is None:
            result._xform_to_j2000_ = Transform.IDENTITY

        return result

    def mask_where(self, mask):
        """A shallow copy of this Event with a new mask, using mask_where."""

        def apply_mask_where(arg):
            if arg.shape != self.shape:
                arg = arg.broadcast_to(self.shape)

            return arg.mask_where(mask)

        result = self._apply_this_func(apply_mask_where)
        return result

    def remask(self, mask):
        """A shallow copy of this Event with a new mask, using remask."""

        def apply_remask(arg):
            if arg.shape != self.shape:
                arg = arg.broadcast_to(self.shape)

            return arg.remask(mask)

        result = self._apply_this_func(apply_remask)
        return result

    def replace(self, *args):
        """A shallow copy with a specific set of attributes replaced.
        """

        pairs = []
        omissions = []
        for k in range(0,len(args),2):
            name = args[k]
            if name in Event.SPECIAL_PROPERTIES:
                if 'arr' in name and '_lt' not in name:
                    omissions.append('arr')
                elif 'dep' in name and '_lt' not in name:
                    omissions.append('dep')
                else:
                    omissions.append(name)
            else:
                omissions.append(name)

            pairs.append((name, args[k+1]))

        result = self.copy(omit=omissions)

        for (name, value) in pairs:
            result.insert_subfield(name, value)

        return result

    ######################################################################################
    # Functions to insert self-derivatives
    ######################################################################################

    def with_time_derivs(self):
        """A clone of this event containing unit time derivatives d_dt in the
        frame of the event.

        Note that the time derivatives of the line of sight are always included
        automatically, based on the time-dependence of the transform to J2000.
        """

        if 't' in self._time_.derivs:
            return self

        event = self.copy()
        event._time_.insert_deriv('t', Scalar.ONE, override=True)

        if (event._ssb_ is not None and event._ssb_ is not event
            and event._ssb_._time_ is not event._time_):
            event.ssb._time_.insert_deriv('t', Scalar.ONE, override=True)

        return event

    def with_los_derivs(self):
        """A clone of this event with unit photon arrival derivatives d_dlos."""

        if 'los' in self.neg_arr_ap.derivs:
            return self

        event = self.copy(omit='arr')

        neg_arr_ap = self.neg_arr_ap.unit().copy()
        neg_arr_ap.insert_deriv('los', Vector3.IDENTITY, override=True)
        event.neg_arr_ap = neg_arr_ap

        return event

    def with_pos_derivs(self):
        """A clone of this event with unit position derivatives d_dpos.

        The derivatives are with respect to the position in the frame of the event.

        Returns:
            Event: The clone, or this event unchanged if it already carries the
            derivatives.
        """

        if 'pos' in self._state_.derivs:
            return self

        event = self.copy()
        event._state_.insert_deriv('pos', Vector3.IDENTITY, override=True)

        if event._ssb_ is not None and event._ssb_ is not event:
            # Rotating the state carries its derivative along, and the rotated derivative
            # is what the SSB version needs; the rotated state itself is a position.
            rotated = event.xform_to_j2000.rotate(event._state_, derivs=True)
            event.ssb._state_.insert_deriv('pos', rotated.d_dpos, override=True)

        return event

    def with_lt_derivs(self):
        """A clone of this event with unit photon arrival light-time derivatives d_dlt.

        Returns:
            Event: The clone, or this event unchanged if it already carries the
            derivatives.
        """

        if 'lt' in self.arr_lt.derivs:
            return self

        event = self.copy()
        event._arr_lt_.insert_deriv('lt', Scalar.ONE, override=True)

        if (event._ssb_ is not None and event._ssb_ is not event
            and event._ssb_._arr_lt_ is not event._arr_lt_):
            event._ssb_._arr_lt_.insert_deriv('lt', Scalar.ONE, override=True)

        return event

    def with_dep_derivs(self):
        """A clone of this event with unit photon departure derivatives d_ddep.

        Returns:
            Event: The clone, or this event unchanged if it already carries the
            derivatives.
        """

        if 'dep' in self.dep_ap.derivs:
            return self

        event = self.copy(omit='dep')

        dep_ap = self.dep_ap.copy()
        dep_ap.insert_deriv('dep', Vector3.IDENTITY, override=True)
        event.dep_ap = dep_ap

        return event

    def with_dlt_derivs(self):
        """A clone of this event with unit photon departure light-time derivatives d_ddlt.

        Returns:
            Event: The clone, or this event unchanged if it already carries the
            derivatives.
        """

        if 'dlt' in self.dep_lt.derivs:
            return self

        event = self.copy()
        event._dep_lt_.insert_deriv('dlt', Scalar.ONE, override=True)

        if (event._ssb_ is not None and event._ssb_ is not event
            and event._ssb_._dep_lt_ is not event._dep_lt_):
            event._ssb_._dep_lt_.insert_deriv('dlt', Scalar.ONE, override=True)

        return event

    ######################################################################################
    # Shrink and unshrink operations
    ######################################################################################

    def shrink(self, antimask):
        """A shrunken version of this event.

        Parameters:
            antimask (Boolean, bool, or None): None to leave the Event unchanged;
                otherwise True where values are kept and False where they are ignored. A
                single value of True keeps everything and a single value of False ignores
                everything.

        Returns:
            Event: The shrunken Event, or this Event where `antimask` keeps everything.
        """

        def shrink1(arg):
            return arg.shrink(antimask)

        if antimask is None:
            return self
        if Qube.is_one_true(antimask):
            return self

        result = self._apply_this_func(shrink1)

        if self._xform_to_j2000_ is not None:
            xform = self._xform_to_j2000_
            new_xform = Transform(xform.matrix.shrink(antimask),
                                  xform.omega.shrink(antimask),
                                  xform.frame, xform.reference, origin=xform.origin)
            result._xform_to_j2000_ = new_xform

        ssb = result._ssb_
        if (ssb is not None and ssb is not result):
            ssb._xform_to_j2000_ = Transform.IDENTITY

        return result

    def unshrink(self, antimask, *, shape=None):
        """Expand a shrunken version of this event to its original state.

        Parameters:
            antimask (Boolean, bool, or None): None to leave the Event unchanged;
                otherwise the boolean array whose True values were kept by `shrink`.
            shape (tuple, optional): Shape to restore; default None to infer it from
                `antimask`.

        Returns:
            Event: The expanded Event, masked wherever `antimask` is False.
        """

        def unshrink1(arg, mask):
            if arg.shape:
                arg = arg.remask(mask)
            return arg.unshrink(antimask, shape=shape)

        # Make sure the new mask applies everywhere
        if antimask is None or Qube.is_one_true(antimask):
            return self.remask(self.mask)

        result = self._apply_this_func(unshrink1, self.mask)

        if self._xform_to_j2000_ is not None:
            xform = self._xform_to_j2000_
            new_xform = Transform(xform.matrix.unshrink(antimask, shape=shape),
                                  xform.omega.unshrink(antimask, shape=shape),
                                  xform.frame, xform.reference, origin=xform.origin)
            result._xform_to_j2000_ = new_xform

        ssb = result._ssb_
        if (ssb is not None and ssb is not result):
            ssb._xform_to_j2000_ = Transform.IDENTITY

        return result

    ######################################################################################
    # Event transformations
    ######################################################################################

    def wrt_ssb(self, *, derivs=True, quick=None):
        """This event relative to SSB coordinates in the J2000 frame.

        This value is cached inside of the object so it can be quickly accessed again at a
        later time.

        Parameters:
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if self._ssb_ is not None:
            if derivs:
                return self._ssb_
            else:
                return self._ssb_.wod

        if self._origin_ == Event.SSB and self._frame_ == Frame.J2000:
                self._ssb_ = self
                self._ssb_._ssb_ = self
                self._xform_to_j2000_ = Transform.identity(Frame.J2000)
                if derivs:
                    return self._ssb_
                else:
                    return self._ssb_.wod

        (self._ssb_,
         self._xform_to_j2000_) = self.wrt(Event.SSB, Frame.J2000, derivs=derivs,
                                           quick=quick, include_xform=True)

        if self._ssb_ is not self:
            self._ssb_._ssb_ = self._ssb_
            self._ssb_._xform_to_j2000_ = Transform.IDENTITY

        if derivs:
            return self._ssb_
        else:
            return self._ssb_.wod

    def from_ssb(self, path, frame, *, derivs=True, quick=None):
        """This SSB/J2000-relative event to a new path and frame.

        Parameters:
            path (Path): Or path ID identifying the new origin; None to leave the origin
                unchanged.
            frame (Frame): Or frame ID of the new coordinate frame; None to leave the
                frame unchanged.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if self._frame_ != Frame.J2000 or self._origin_ != Event.SSB:
            raise ValueError('Event.from_ssb requires a SSB/J2000 event')

        event = self.wrt(path, frame, derivs=True, quick=quick)
        event._ssb_ = self
        event._ssb_._ssb_ = self

        if derivs:
            return event
        else:
            return event.wod

    def wrt(self, path=None, frame=None, *, derivs=True, quick=None, include_xform=False):
        """This event relative to a new path and/or a new coordinate frame.

        Parameters:
            path (Path or str, optional): The new origin path or its ID; None to leave the
                origin unchanged.
            frame (Frame or str, optional): The new coordinate frame or its ID; None to
                leave the frame unchanged.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            include_xform (bool, optional): If True, the transform is returned in a tuple
                along with the new event.

        Returns:
            Event or tuple[Event, Transform]: The new Event; if `include_xform` is True, a
            tuple of the new Event and the Transform from this event's frame to the new
            frame.
        """

        # Interpret inputs
        if path is None:
            path = self._origin_
        else:
            path = Event._Path.as_path(path)

        if frame is None:
            frame = self._frame_
        else:
            frame = Frame.as_frame(frame)

        # Point to the working copy of this Event object
        event = self

        # If the path is shifting...
        xform1 = None
        if event._origin_.waypoint != path.waypoint:

            # ...and the current frame is rotating...
            old_frame = event._frame_
            if old_frame.origin is not None:

                # ...then rotate to J2000
                (event, xform1) = event.wrt_frame(Frame.J2000,
                                                  derivs=derivs, quick=quick,
                                                  include_xform=True)

        # If the frame is changing...
        if event._frame_.wayframe != frame.wayframe:

            # ...and the new frame is rotating...
            if frame.origin is not None:

                # ...then shift to the origin of the new frame
                event = event.wrt_path(frame.origin, derivs=derivs, quick=quick)

        # Now it is safe to rotate to the new frame
        (event, xform) = event.wrt_frame(frame, derivs=derivs, quick=quick,
                                         include_xform=True)

        # Now it is safe to shift to the new event
        event = event.wrt_path(path, derivs=derivs, quick=quick)

        # Now fix the frame again if necessary
        xform2 = None
        if event._frame_.wayframe != frame.wayframe:
            (event, xform2) = event.wrt_frame(frame, derivs=derivs, quick=quick,
                                              include_xform=True)

        # Return results
        if include_xform:
            xform = xform.rotate_transform(xform1) if xform1 else xform
            xform = xform2.rotate_transform(xform) if xform2 else xform
            return (event, xform)
        else:
            return event

    def wrt_path(self, path, *, derivs=True, quick=None):
        """This event defined relative to a different origin path.

        The frame is unchanged.

        Parameters:
            path (Path): Object to be used as the new origin. If the value is None, the
                event is returned unchanged.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if path is None:
            path = self._origin_
        else:
            path = Event._Path.as_path(path)

        if self._origin_.waypoint == path.waypoint:
            if derivs:
                return self
            else:
                return self.wod

        # Make sure frames match; make recursive calls to wrt() if needed
        event = self
        if self.frame.wayframe != path.frame.wayframe:
            event = event.wrt(path, path.frame, derivs=derivs, quick=quick)

        new_path = event._origin_.wrt(path, path.frame)
        result = new_path.add_to_event(event, derivs=derivs, quick=quick)

        # Other attributes do not depend on the path
        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = event.__dict__[attr]

        for (name, value) in event._subfields_.items():
            result.insert_subfield(name, value)

        if derivs:
            return result
        else:
            return result.wod

    def wrt_frame(self, frame, *, derivs=True, quick=None, include_xform=False):
        """This event defined relative to a different frame.

        The path is unchanged.

        Parameters:
            frame (Frame): Object to be used as the new reference. If the value is None,
                the event is returned unchanged.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            include_xform (bool, optional): If True, the transform is returned in a tuple
                along with the new event.
        """

        if frame is None:
            frame = self._frame_
        else:
            frame = Frame.as_frame(frame)

        if self._frame_.wayframe == frame.wayframe:
            if derivs:
                result = self
            else:
                result = self.wod

            if include_xform:
                return (result, Transform.identity(frame))
            else:
                return result

        new_frame = frame.wrt(self._frame_)
        return self.rotate_by_frame(new_frame, derivs=derivs, quick=quick,
                                               include_xform=include_xform)

    def rotate_by_frame(self, frame, *, derivs=True, quick=None, include_xform=False):
        """This event rotated forward into a new frame.

        The origin is unchanged. Subfields are also rotated into the new frame.

        Parameters:
            frame (Frame): Into which to transform the coordinates. Its reference frame
                must be the current frame of the event.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            include_xform (bool, optional): If True, the transform is returned in a tuple
                along with the new event.
        """

        def xform_rotate(arg):
            try:
                return xform.rotate(arg, derivs=True)
            except (ValueError, TypeError, KeyError):
                return arg

        if derivs:
            event = self
        else:
            event = self.wod

        frame = Frame.as_frame(frame)
        xform = frame.transform_at_time(event._time_, quick=quick)
        # xform rotates from event frame to new frame

        state = xform.rotate(event._state_, derivs=True)

        result = Event(event._time_, state, event._origin_, frame.wayframe)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = xform_rotate(event.__dict__[attr])

        result._xform_to_j2000_ = None

        for (name, value) in event._subfields_.items():
            result.insert_subfield(name, xform_rotate(value))

        if include_xform:
            return (result, xform)
        else:
            return result

    def unrotate_by_frame(self, frame, *, derivs=True, quick=None):
        """This Event unrotated back into the given frame.

        The origin is unchanged. Subfields are also unrotated.

        Parameters:
            frame (Frame): Object to inverse-transform the coordinates. Its target frame
                must be the current frame of the event. The returned event will use the
                reference frame instead.
            derivs (bool, optional): True to include the derivatives in the returned
                Event; False to exclude them. Time derivatives are always retained.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        def xform_unrotate(arg):
            try:
                return xform.unrotate(arg, derivs=True)
            except (ValueError, TypeError, KeyError):
                return arg

        if derivs:
            event = self
        else:
            event = self.wod

        frame = Frame.as_frame(frame)
        xform = frame.transform_at_time(event._time_, quick=quick)

        state = xform.unrotate(event._state_, derivs=True)

        result = Event(event._time_, state, event._origin_, frame.reference)

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            result.__dict__[attr] = xform_unrotate(event.__dict__[attr])

        result._xform_to_j2000_ = None

        for (name, value) in event._subfields_.items():
            result.insert_subfield(name, xform_unrotate(value))

        return result

    def collapse_time(self, threshold=None):
        """If the time span is small, return a similar Event having fixed time.

        If the difference between the earliest time and the latest time is smaller than a
        specified threshold, a new Event object is returned in which the time is replaced
        by a Scalar equal to the midtime.

        Otherwise, the object is returned unchanged.

        Parameters:
            threshold (float, optional): The allowed difference in seconds between the
                earliest and latest times. None to use the value specified by the
                EVENT_CONFIG.
        """

        def without_derivs(arg):
            if arg is None:
                return arg
            return arg.wod

        if self._time_.shape == ():
            return self
        if self._time_.derivs:
            return self

        if threshold is None:
            threshold = EVENT_CONFIG.collapse_threshold

        tmin = self._time_.min()
        tmax = self._time_.max()
        span = tmax - tmin

        collapsed_mask = (span == Scalar.MASKED)

        if span > threshold:
            return self

        if LOGGING.event_time_collapse:
            LOGGING.diagnostic('Event.collapse_time()', tmin, tmax - tmin)

        midtime = Scalar((tmin + tmax)/2., collapsed_mask, self._time_.units)

        result = self.copy()
        result._time_ = midtime

        if result._ssb_ is not None and result._ssb_ is not result:
            result._ssb_._time_ = midtime

        result._shape_ = None

        return result

    ######################################################################################
    # Event subtraction
    ######################################################################################

    def sub(self, reference, *, quick=None):
        """The result of subtracting the reference event from this event.

        Used mainly for debugging. Note that the reference event could occur at a
        different time. Subtracted events can only be used to examine differences in
        properties; other methods will fail.

        Vectors in the returned object are in the frame of the reference event; times are
        relative. Photon events are unchanged except for the coordinate transform.

        The returned object has additional attributes 'event' and 'reference', which point
        to the source events.

        Parameters:
            reference (Event): The Event to subtract from this one.
            quick (dict or bool, optional): Overrides for the QuickPath and QuickFrame
                parameters; use False to disable them. Default None applies the values in
                the QUICK configuration.

        Returns:
            Event: The difference, with vectors in the frame of `reference` and times
            relative to it.
        """

        def ref_unrotate(arg):
            try:
                return reference.xform_to_j2000.unrotate(arg)
            except (ValueError, TypeError, KeyError):
                return arg

        event_ssb = self.wrt_ssb(derivs=True, quick=quick)
        reference_ssb = reference.wrt_ssb(derivs=True, quick=quick)

        time = self.time - reference.time
        state = ref_unrotate(event_ssb.state - reference_ssb.state)
        diff = Event(time, state, reference.origin, reference.frame)

        diff._ssb_ = self._ssb_

        for prop_name in Event.SPECIAL_PROPERTIES:
            attr = Event.attr_name(prop_name)
            diff.__dict__[attr] = ref_unrotate(event_ssb.__dict__[attr])

        for (key,subfield) in event_ssb._subfields_.items():
            try:
                subfield = ref_unrotate(subfield)
            except (ValueError, TypeError, KeyError):
                pass

            diff.insert_subfield(key, subfield)

        diff.event = self
        diff.reference = reference

        return diff

    ######################################################################################
    # Aberration procedures
    ######################################################################################

    def apparent_ray_ssb(self, ray_ssb, *, derivs=False, quick=None):
        """Apparent direction of a photon in the SSB/J2000 frame. Not cached.

        Parameters:
            ray_ssb (Vector3): The true direction of a light ray in the SSB/J2000 system
                (not reversed!).
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # This procedure is equivalent to a vector subtraction of the velocity of the
        # observer from the ray, given that the ray has length C. However the length of
        # the ray is adjusted to be accurate to higher order in (v/c)

        ray_ssb = Vector3.as_vector3(ray_ssb, recursive=derivs).as_readonly()
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        vel_ssb = wrt_ssb.vel + wrt_ssb.vflat.without_deriv('t')

        # Below, factor = 1 is good to first order, matching the accuracy of the SPICE
        # toolkit. The expansion in beta below was determined empirically to match the
        # exact expression for the aberration, which is:
        #   tan(alpha'/2) = sqrt((1+beta)/(1-beta)) tan(alpha/2)
        #
        # alpha is the actual angle between the velocity vector and the photon's direction
        # of motion (NOT reversed).
        #
        # alpha' is the apparent angle.

        beta = C_INVERSE * vel_ssb.norm()
        ray_ssb_norm = ray_ssb.norm()
        cos_angle = (C_INVERSE * vel_ssb.dot(ray_ssb)
                     / (ray_ssb_norm * beta))
        factor = 1. - beta * (cos_angle - beta) * (0.5 + 0.375 * beta**2)

        return ray_ssb - (factor * C_INVERSE) * ray_ssb_norm * vel_ssb

    def actual_ray_ssb(self, ray_ap_ssb, *, derivs=False, quick=None):
        """Actual direction of a photon in the SSB/J2000 frame. Not cached.

        Parameters:
            ray_ap_ssb (Vector3): The apparent direction of a light ray in the SSB/J2000
                system.
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # This procedure is equivalent to a vector subtraction of the velocity
        # of the observer from the ray, given that the ray has length C.

        ray_ap_ssb = Vector3.as_vector3(ray_ap_ssb, recursive=derivs).as_readonly()
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        vel_ssb = wrt_ssb.vel + wrt_ssb.vflat.without_deriv('t')

        # Invert the function above
        beta_ssb = C_INVERSE * vel_ssb
        beta = beta_ssb.norm()
        vel_inv = C_INVERSE / beta
        bb = beta * (0.5 + 0.375 * beta**2)
        f1 = 1. + bb * beta

        # Iterate solution
        ITERS = 4
        ray_ssb = ray_ap_ssb
        for count in range(ITERS):
            ray_ssb_norm = ray_ssb.norm()
            cos_angle = vel_inv * vel_ssb.dot(ray_ssb) / ray_ssb_norm
            factor = f1 - bb * cos_angle
            ray_ssb = ray_ap_ssb + factor * ray_ssb_norm * beta_ssb

        return ray_ssb

    def apparent_arr(self, *, derivs=False, quick=None):
        """Apparent direction of an arriving ray in the event frame. Cached.

        Parameters:
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if self._arr_ap_ is not None:
            if derivs:
                return self._arr_ap_
            else:
                return self._arr_ap_.wod

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        arr_ap_ssb = self.apparent_ray_ssb(wrt_ssb.arr, derivs=derivs, quick=quick)
        wrt_ssb._arr_ap_ = arr_ap_ssb

        # Convert to this event's frame
        if self._frame_ != Frame.J2000:
            self._arr_ap_ = self._xform_to_j2000_.unrotate(arr_ap_ssb,
                                                           derivs=True)
        else:
            self._arr_ap_ = arr_ap_ssb

        # Cache the result
        if derivs:
            return self._arr_ap_
        else:
            return self._arr_ap_.wod

    def actual_arr(self, *, derivs=False, quick=None):
        """Actual direction of an arriving ray in the event frame. Cached.

        Parameters:
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if self._arr_ is not None:
            if derivs:
                return self._arr_
            else:
                return self._arr_.wod

        # Otherwise, calculate and cache the actual vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        arr_ssb = self.actual_ray_ssb(wrt_ssb.arr_ap, derivs=derivs, quick=quick)
        wrt_ssb._arr_ = arr_ssb

        # Convert to this event's frame
        if self._frame_ != Frame.J2000:
            self._arr_ = self._xform_to_j2000_.unrotate(arr_ssb, derivs=True)
        else:
            self._arr_ = arr_ssb

        # Cache the result
        if derivs:
            return self._arr_
        else:
            return self._arr_.wod

    def apparent_dep(self, *, derivs=False, quick=None):
        """Apparent direction of a departing ray in the event frame. Cached.

        Parameters:
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if self._dep_ap_ is not None:
            if derivs:
                return self._dep_ap_
            else:
                return self._dep_ap_.wod

        # Otherwise, calculate and cache the apparent vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        dep_ap_ssb = self.apparent_ray_ssb(wrt_ssb._dep_, derivs=derivs, quick=quick)
        wrt_ssb._dep_ap_ = dep_ap_ssb

        # Convert to this event's frame
        if self._frame_ != Frame.J2000:
            self._dep_ap_ = self._xform_to_j2000_.unrotate(dep_ap_ssb,
                                                           derivs=True)
        else:
            self._dep_ap_ = dep_ap_ssb

        # Cache the result
        if derivs:
            return self._dep_ap_
        else:
            return self._dep_ap_.wod

    def actual_dep(self, *, derivs=False, quick=None):
        """Actual direction of a departing ray in the event frame. Cached.

        Parameters:
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned ray; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        # If the apparent vector is already cached, return it
        if self._dep_ is not None:
            if derivs:
                return self._dep_
            else:
                return self._dep_.wod

        # Otherwise, calculate and cache the actual vector in the SSB frame
        wrt_ssb = self.wrt_ssb(derivs=derivs, quick=quick)
        dep_ssb = self.actual_ray_ssb(wrt_ssb._dep_ap_, derivs=derivs, quick=quick)
        wrt_ssb._dep_ = dep_ssb

        # Convert to this event's frame
        if self._frame_ != Frame.J2000:
            self._dep_ = self._xform_to_j2000_.unrotate(dep_ssb, derivs=True)
        else:
            self._dep_ = dep_ssb

        # Cache the result
        if derivs:
            return self._dep_
        else:
            return self._dep_.wod

    def incidence_angle(self, apparent=False, *, derivs=False, quick=None):
        """The incidence angle.

        The incidence angle is measured between the surface normal and the reversed
        direction of the arriving photon.

        Parameters:
            apparent (bool, optional): True to account for the aberration in the Event
                frame.
            derivs (bool, optional): True to include the derivatives of the light ray in
                the returned angle; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if self._arr_ is None and self._arr_ap_ is None:
            raise ValueError(f'undefined arrival vector in {self}')

        if self._perp_ is None:
            raise ValueError(f'undefined perpendicular vector in {self}')

        shrunk = self.shrink(self.antimask)
        _ = shrunk.wrt_ssb(derivs=True, quick=quick)

        if apparent:
            arr = shrunk.arr_ap
        else:
            arr = shrunk.arr

        result = Scalar.PI - shrunk.perp.sep(arr, recursive=derivs)
        return result.unshrink(self.antimask, shape=self.shape)

    def emission_angle(self, apparent=False, *, derivs=False, quick=None):
        """The emission angle.

        The emission angle is measured between the surface normal and the direction of the
        departing photon.

        Parameters:
            apparent (bool, optional): True to account for the aberration in the Event
                frame.
            derivs (bool, optional): True to include any derivatives of the light ray in
                the returned angle; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if self._dep_ is None and self._dep_ap_ is None:
            raise ValueError(f'undefined departure vector in {self}')

        if self._perp_ is None:
            raise ValueError(f'undefined perpendicular vector in {self}')

        shrunk = self.shrink(self.antimask)
        _ = shrunk.wrt_ssb(derivs=True, quick=quick)

        if apparent:
            dep = shrunk.dep_ap
        else:
            dep = shrunk.dep

        result = shrunk.perp.sep(dep, recursive=derivs)
        return result.unshrink(self.antimask, shape=self.shape)

    def phase_angle(self, apparent=False, *, derivs=False, quick=None):
        """The phase angle.

        The phase angle is measured between the apparent direction of the arriving photon
        and the reversed direction of the departing photon.

        Parameters:
            apparent (bool, optional): True to account for the aberration in the Event
                frame.
            derivs (bool, optional): True to include any derivatives of the light ray in
                the returned angle; False to exclude them.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
        """

        if self._arr_ is None and self._arr_ap_ is None:
            raise ValueError(f'undefined arrival vector in {self}')

        if self._dep_ is None and self._dep_ap_ is None:
            raise ValueError(f'undefined departure vector in {self}')

        shrunk = self.shrink(self.antimask)
        _ = shrunk.wrt_ssb(derivs=True, quick=quick)

        if apparent:
            dep = shrunk.dep_ap
            arr = shrunk.arr_ap
        else:
            dep = shrunk.dep
            arr = shrunk.arr

        result = Scalar.PI - dep.sep(arr, recursive=derivs)
        return result.unshrink(self.antimask, shape=self.shape)

    def ra_and_dec(self, apparent=False, *, derivs=False, subfield='arr', quick=None,
                   frame='J2000'):
        """The right ascension and declination as a tuple of two Scalars.

        Parameters:
            apparent (bool, optional): True to include stellar aberration, thereby
                returning the apparent direction of the photon relative to the background
                stars; False to return the purely geometric values, neglecting the motion
                of the observer.
            derivs (bool, optional): True to include any derivatives of the light ray in
                the returned quantities; False to exclude them.
            subfield (optional): The subfield to use for the calculation, either "arr"
                or "dep". Note that an arriving direction is reversed.
            quick (dict, optional): To override the configured default parameters for
                QuickPaths and QuickFrames; False to disable the use of QuickPaths and
                QuickFrames. The default configuration is defined in config.py.
            frame (Frame, optional): Coordinate frame for RA and dec. Default is J2000.
                Use None to use the frame of this event.
        """

        # Validate the inputs
        if subfield not in ('arr', 'dep'):
            raise ValueError(f'invalid input value for subfield: {subfield!r}')

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
        else:
            if subfield == 'arr':
                ray = event.neg_arr_ap
            else:
                ray = event.dep_ap

        if ray is None:
            raise ValueError(f'undefined light ray vector in {self}')

        # Convert to RA and dec
        return ray.to_ra_dec_length(recursive=derivs)[:2]

##########################################################################################
