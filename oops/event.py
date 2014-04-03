################################################################################
# oops/event.py: Event class
################################################################################

import numpy as np
from polymath import *

from oops.frame_.frame import Frame
from oops.config       import EVENT_CONFIG, LOGGING
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

        __subfields_ a dictionary of PolyMath objects providing further
                    information about the properties of the event.

    Each of these attributes can also be accessed via a read-only property with
    the same name except for the surrounding undercores.

    Subfields are accessible both via the __subfields_ dictionary, and also
    directly as attributes of the Event object. These are the most common
    subfields:

        arr         an optional Vector3 defining the direction of a photon
                    arriving at this event. It is defined in the coordinate
                    frame of this event and not corrected for stellar
                    aberration. Length is arbitrary.

        dep         an optional Vector3 defining the direction of a photon
                    departing from this event. It is defined in the coordinate
                    frame of this event and not corrected for stellar
                    aberration. Length is arbitrary.

        arr_lt      an optional Scalar defining the (negative) light travel time
                    for the arriving photon from its origin.

        dep_lt      an optional Scalar defining the light travel time of a
                    departing photon to its destination.

        perp        the direction of a normal vector if this event falls on a
                    surface.

        vflat       a velocity component within the surface, which can be used
                    to describe winds across a planet or orbital motion
                    within a ring plane.

    Note that the attributes of an object need not have the same shape, but they
    must all be broadcastable to the same shape.

    Events are intended to be static, with the exception that users might have
    reasons to modify some subfields.
    """

    PATH_CLASS = None       # undefined at load to avoid circular dependencies

    def __init__(self, time, state, origin, frame=None, **subfields):
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
            **subfields an arbitrary set of subfields that are will also be
                        accessible as attributes of the Event object.
        """

        self.__time_ = Scalar.as_scalar(time).as_readonly()

        if type(state) in (tuple,list) and len(state) == 2:
            pos = Vector3.as_vector3(state[0]).as_readonly()
            vel = Vector3.as_vector3(state[1]).as_readonly()
            pos.insert_deriv('t', vel, override=True)
            self.__state_ = pos
        else:
            self.__state_ = Vector3.as_vector3(state).as_readonly()

        origin = Event.PATH_CLASS.as_primary_path(origin)
        self.__origin_ = origin.waypoint
        self.__frame_  = origin.frame.wayframe

        if frame is not None:
            self.__frame_ = Frame.as_wayframe(frame)

        # Fill in default values for subfields as attributes
        self.dep = Empty.EMPTY
        self.arr = Empty.EMPTY
        self.dep_lt = Empty.EMPTY
        self.arr_lt = Empty.EMPTY
        self.perp = Empty.EMPTY
        self.vflat = Vector3.ZERO

        # Overwrite with given subfields
        self.__subfields_ = {}
        for (key,subfield) in subfields.iteritems():
            self.insert_subfield(key, subfield.as_readonly())

        # Used if needed
        self.filled_ssb = None
        self.filled_shape = None
        self.filled_mask = None

    @property
    def time(self):
        return self.__time_

    @property
    def state(self):
        return self.__state_

    @property
    def pos(self):
        return self.__state_.without_derivs()

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
        if self.filled_shape is None:
            self.filled_shape = Qube.broadcasted_shape(self.__time_,
                                                       self.__state_,
                                                       self.__origin_,
                                                       self.__frame_)
        return self.filled_shape

    @property
    def mask(self):
        if self.filled_mask is None:
            self.filled_mask = (self.__time_.mask | self.__state_.mask |
                                                    self.vel.mask)

        return self.filled_mask

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

        keys = self.subfields.keys()
        keys.sort()
        for key in keys:
            str_list += ['; ', key]

        str_list += [')']
        return ''.join(str_list)

    def insert_subfield(self, key, value):
        """Insert a given subfield into this Event."""

        self.__subfields_[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

        self.filled_ssb = None          # SSB version is now out of date

    def insert_self_derivs(self, override=False):
        """Insert unit derivatives into the time and position attributes.

        The derivatives are time.d_dt_link and state.d_dpos_link. They can be
        used to track derivatives among events that link with this one.
        """

        self.__time_.insert_deriv('t_link', Scalar.ONE, override=override)
        self.__state_.insert_deriv('pos_link', Vector3.IDENTITY,
                                               override=override)
        return self

    def clone(self, subfields=True, recursive=False):
        """A shallow copy of the Event.

        Inputs:
            subfields   True to transfer the subfields.
            recursive   True also to clone (shallow-copy) the attributes of the
                        Event. This is necessary if derivatives of the subfields
                        are going to be modified.
        """

        if recursive:
            result = Event(self.__time_.clone(), self.__state_.clone(),
                           self.__origin_, self.__frame_)
        else:
            result = Event(self.__time_, self.__state_,
                           self.__origin_, self.__frame_)

        if subfields:
            for (key,subfield) in self.__subfields_.iteritems():
                if recursive: subfield = subfield.clone()
                result.insert_subfield(key, subfield)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask

        if self.filled_ssb is self:
            result.filled_ssb = result
        else:
            result.filled_ssb = None

        return result

    def without_derivs(self):
        """A shallow copy of this Event without derivatives.

        Input:
            subfields   True to retain subfields; False to remove them.
        """

        result = Event(self.__time_.without_derivs(),
                       self.__state_.without_derivs(preserve='t'),
                       self.__origin_, self.__frame_)

        for (key,subfield) in self.subfields.iteritems():
            result.insert_subfield(key, subfield.without_derivs(preserve='t'))

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask

        if self.filled_ssb is self:
            result.filled_ssb = result
        else:
            result.filled_ssb = None

        return result

    def without_mask(self):
        """A shallow copy of this Event with the masks on subfields removed."""

        result = Event(self.__time_.without_mask(recursive),
                       self.__state_.without_mask(recursive),
                       self.__origin_, self.__frame_)

        for (key,subfield) in self.subfields.iteritems():
            result.insert_subfield(key, subfield.without_mask(recursive))

        result.filled_shape = self.filled_shape
        result.filled_mask  = False

        if self.filled_ssb is self:
            result.filled_ssb = result
        else:
            result.filled_ssb = None

        return result

    def all_masked(self, origin=None, frame=None, derivs=True):
        """A shallow copy of this event, entirely masked.

        Inputs:
            origin      the origin or origin_id of the Event returned; if None,
                        use the origin of this Event.
            frame       the frame or frame_id of the Event returned; if None,
                        use the frame of this Event.
            derivs      True to include derivatives in the returned Event.
        """

        if origin is None:
            origin = self.__origin_
        else:
            origin = Event.PATH_CLASS.as_waypoint(origin)

        if frame is None:
            frame = self.__frame_
        else:
            frame = Frame.as_wayframe(frame)

        event = Event(self.__time_.all_masked(derivs),
                      self.__state_.all_masked(derivs),
                      origin, frame)

        for (key,subfield) in self.subfields.iteritems():
            event.insert_subfield(key, subfield.all_masked(derivs))

        # Because this is masked, we can re-use this Event for the SSB version
        event.filled_ssb = event.clone()
        event.filled_ssb.__origin_ = Event.PATH_CLASS.SSB
        event.filled_ssb.__frame_ = Frame.J2000

        event.filled_mask = True
        return event

    ############################################################################
    # Event transformations
    ############################################################################

    def wrt_ssb(self, derivs=True, quick=False):
        """This event relative to SSB coordinates in the J2000 frame.

        This value is cached inside of the object so it can be quickly accessed
        again at a later time.

        Input:
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        if self.filled_ssb is None:
            self.filled_ssb = self.wrt(Event.PATH_CLASS.SSB, Frame.J2000,
                                       derivs=derivs, quick=quick)
            self.filled_ssb.filled_ssb = self.filled_ssb

        if derivs:
            return self.filled_ssb
        else:
            return self.filled_ssb.without_derivs()

    def wrt(self, path=None, frame=None, derivs=True, quick=False):
        """This event relative to a new path and/or a new coordinate frame.

        Input:
            path        the Path or path ID identifying the new origin;
                        None to leave the origin unchanged.
            frame       the Frame or frame ID of the new coordinate frame; None
                        to leave the frame unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.

        The new derivatives will multiply the originals, yielding derivatives
        with respect to the same event as those of this event.
        """

        # Interpret inputs
        if path is None: path = self.__origin_
        if frame is None: frame = self.__frame_

        path = Event.PATH_CLASS.as_path(path)
        frame = Frame.as_frame(frame)

        # Point to the working copy of this Event object
        event = self

        # If the path is shifting...
        if event.__origin_.waypoint != path.waypoint:

            # ...and the current frame is rotating...
            old_frame = event.__frame_
            if old_frame.origin is not None:

                # ...then rotate to J2000
                event = event.wrt_frame(Frame.J2000, derivs, quick)

        # If the frame is changing...
        if event.__frame_.wayframe != frame.wayframe:

            # ...and the new frame is rotating...
            if frame.origin is not None:

                # ...then shift to the origin of the new frame
                event = event.wrt_path(frame.origin, derivs, quick)

        # Now it is safe to rotate to the new frame
        event = event.wrt_frame(frame, derivs, quick)

        # Now it is safe to shift to the new path
        event = event.wrt_path(path, derivs, quick)

        return event

    def wrt_path(self, path, derivs=True, quick=False):
        """This event defined relative to a different origin path.

        The frame is unchanged.

        Input:
            path        the Path object to be used as the new origin. If the
                        value is None, the event is returned unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        if path is None: return self

        path = Event.PATH_CLASS.as_path(path)
        if self.__origin_.waypoint == path.waypoint:
            if derivs:
                return self
            else:
                return self.without_derivs()

        new_path = self.__origin_.wrt(path, path.frame)
        return new_path.add_to_event(self, derivs=derivs, quick=quick)

    def wrt_frame(self, frame, derivs=True, quick=False):
        """This event defined relative to a different frame.

        The path is unchanged.

        Input:
            frame       the Frame object to be used as the new reference. If the
                        value is None, the event is returned unchanged.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        if frame is None: return self

        frame = Frame.as_frame(frame)
        if self.__frame_.wayframe == frame.wayframe:
            if derivs:
                return self
            else:
                return self.without_derivs()

        new_frame = frame.wrt(self.__frame_)
        return self.rotate_by_frame(new_frame, derivs=derivs, quick=quick)

    def rotate_by_frame(self, frame, derivs=True, quick=False):
        """This event rotated forward into a new frame.

        The origin is unchanged. Subfields are also rotated into the new frame.

        Input:
            frame       a Frame into which to transform the coordinates. Its
                        reference frame must be the current frame of the event.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        frame = Frame.as_frame(frame)
        assert self.__frame_ == frame.reference

        transform = frame.transform_at_time(self.__time_, quick=quick)

        if derivs:
            state = transform.rotate(self.__state_, derivs)
        else:
            state = transform.rotate_pos_vel(self.pos, self.vel)

        subfields = {}
        for (key,subfield) in self.__subfields_.iteritems():
            subfields[key] = transform.rotate(subfield, derivs)

        result = Event(self.__time_, state, self.__origin_, frame.wayframe,
                       **subfields)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

        return result

    def unrotate_by_frame(self, frame, derivs=True, quick=False):
        """This Event unrotated back into the given frame.

        The origin is unchanged. Subfields are also urotated.

        Input:
            frame       a Frame object to to inverse-transform the coordinates.
                        Its target frame must be the current frame of the event.
                        The returned event will use the reference frame instead.
            derivs      True to include the derivatives in the returned Event;
                        False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        frame = Frame.as_frame(frame)
        assert self.__frame_ == frame.wayframe

        transform = frame.transform_at_time(self.__time_, quick=quick)

        if derivs:
            state = transform.unrotate(self.__state_, derivs)
        else:
            state = transform.unrotate_pos_vel(self.pos, self.vel)

        subfields = {}
        for (key,subfield) in self.__subfields_.iteritems():
            subfields[key] = transform.unrotate(subfield, derivs)

        result = Event(self.__time_, state, self.__origin_, frame.reference,
                       **subfields)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

        return result

    def collapse_time(self, threshold=None, recursive=False):
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
        return Event(midtime, self.__state_, self.__origin_, self.__frame_,
                                   **self.subfields)

    ############################################################################
    # Geometry procedures
    ############################################################################

    def apparent_ray_ssb(self, ray_ssb, derivs=False, quick={}):
        """Apparent direction of a photon in the SSB/J2000 frame.

        Input:
            ray_ssb     the true direction of a light ray in the SSB/J2000
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

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)

        if not derivs:
            ray_ssb = ray_ssb.without_derivs()

        return ray_ssb - (wrt_ssb.vel + wrt_ssb.vflat) * ray_ssb.norm() / C

    def apparent_arr_ssb(self, derivs=False, quick={}):
        """Apparent direction of the arriving ray in the SSB/J2000 frame.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        ray_ssb = self.wrt_ssb(derivs, quick=quick).arr
        return self.apparent_ray_ssb(ray_ssb, derivs, quick=quick)

    def apparent_arr(self, derivs=False, quick={}):
        """Apparent direction of an arriving ray in the event frame.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        arr_ssb = self.apparent_arr_ssb(derivs, quick=quick)

        if self.__frame_ == Frame.J2000: return arr_ssb

        frame = self.__frame_.wrt(Frame.J2000)
        xform = frame.transform_at_time(self.__time_, quick=quick)
        return xform.rotate(arr_ssb)

    def apparent_dep_ssb(self, derivs=False, quick={}):
        """Apparent direction of a departing ray in the SSB/J2000 frame.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        ray_ssb = self.wrt_ssb(derivs, quick=quick).dep
        return self.apparent_ray_ssb(ray_ssb, derivs, quick=quick)

    def apparent_dep(self, derivs=False, quick={}):
        """Apparent direction of a departing ray in the event frame.

        Input:
            derivs      True to include the derivatives of the light ray in the
                        returned ray; False to exclude them.
            quick       an optional dictionary to override the configured
                        default parameters for QuickPaths and QuickFrames; False
                        to disable the use of QuickPaths and QuickFrames. The
                        default configuration is defined in config.py.
        """

        dep_ssb = self.apparent_dep_ssb(derivs, quick=quick)

        if self.__frame_ == Frame.J2000:
            return dep_ssb

        frame = self.__frame_.wrt(Frame.J2000)
        xform = frame.transform_at_time(self.__time_, quick=quick)
        return xform.rotate(dep_ssb)

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

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)

        if apparent:
            arr_ssb = wrt_ssb.apparent_arr_ssb(derivs, quick=quick)
        elif derivs:
            arr_ssb = wrt_ssb.arr
        else:
            arr_ssb = wrt_ssb.arr.without_derivs()

        return np.pi - wrt_ssb.perp.sep(arr_ssb)

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

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)

        if apparent:
            dep_ssb = wrt_ssb.apparent_dep_ssb(derivs, quick=quick)
        elif derivs:
            dep_ssb = wrt_ssb.dep
        else:
            dep_ssb = wrt_ssb.dep.without_derivs()

        return wrt_ssb.perp.sep(dep_ssb, derivs)

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

        wrt_ssb = self.wrt_ssb(derivs, quick=quick)

        if apparent:
            dep_ssb = wrt_ssb.apparent_dep_ssb(derivs, quick=quick)
            arr_ssb = wrt_ssb.apparent_arr_ssb(derivs, quick=quick)
        elif derivs:
            dep_ssb = wrt_ssb.dep
            arr_ssb = wrt_ssb.arr
        else:
            dep_ssb = wrt_ssb.dep.without_derivs()
            arr_ssb = wrt_ssb.arr.without_derivs()

        return np.pi - dep_ssb.sep(arr_ssb, derivs)

    def ra_and_dec(self, apparent=False, derivs=False,
                         subfield='arr', quick={}):
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
        """

        # Validate the inputs
        assert subfield in {'arr', 'dep'}

        # Calculate the ray in J2000
        if apparent:
            if subfield == 'dep':
                ray_ssb = self.wrt_ssb().apparent_dep_ssb(derivs, quick=quick)
            else:
                ray_ssb = -self.wrt_ssb().apparent_arr_ssb(derivs, quick=quick)
        elif derivs:
            if subfield == 'dep':
                ray_ssb = self.wrt_ssb().dep
            else:
                ray_ssb = -self.wrt_ssb().arr
        else:
            if subfield == 'dep':
                ray_ssb = self.wrt_ssb().dep.without_derivs()
            else:
                ray_ssb = -self.wrt_ssb().arr.without_derivs()

        # Convert to RA and dec
        (x,y,z) = ray_ssb.to_scalars(derivs)
        ra = y.arctan2(x,derivs) % TWOPI

        r = ray_ssb.norm(derivs)
        dec = (z/r).arcsin(derivs)

        return (ra, dec)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Event(unittest.TestCase):

    def runTest(self):

        # Tested thoroughly in other modules
        pass

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
