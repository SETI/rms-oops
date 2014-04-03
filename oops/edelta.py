################################################################################
# oops/edelta.py: Event-delta class
################################################################################

from polymath import *

from oops.event        import Event
from oops.frame_.frame import Frame
from oops.path_.path   import Path
import oops.constants  as constants

class Edelta(object):
    """An Edelta object is defined by a time, position and velocity as offsets
    relative to another event. Note that the times of the two events can differ.
    """

    def __init__(self, time, state, event, **subfields):
        """Constructor for the Edelta class.

        Input:
            time        a Scalar of event times in seconds TDB.
            state       position vectors as a Vector3 object. The velocity
                        should be included as the time-derivative. However, if
                        specified as a tuple of two objects, the first is
                        interpreted as the position and the second as the
                        velocity.
            event       the event relative to which this event is defined.
            **subfields an arbitrary set of subfields that are will also be
                        accessible as attributes of the Edelta object. They have
                        the same names and meanings as those in the Event class.

        These attributes are defined by the constructor, based on the origin
        event.
            epoch       the time of the origin event.
            origin      the path waypoint defining the origin of this event.
            frame       the wayframe identifying the coordinate system.

        These are defined as read-only properties.
            shape       a tuple of integers defining the overall shape of the
                        event, found as a result of broadcasting together the
                        time, position, velocity, arr and dep attributes.
            mask        the overall mask of the Edelta, constructed as the "or"
                        of the time, pos, vel, arr and dep masks.
        """

        self.time = Scalar.as_scalar(time).as_readonly()

        if type(state) in (tuple,list) and len(state) == 2:
            pos = Vector3.as_vector3(state[0]).as_readonly()
            vel = Vector3.as_vector3(state[1]).as_readonly()
            pos.insert_deriv('t', vel, override=True)
            self.state = pos
        else:
            self.state = Vector3.as_vector3(state).as_readonly()

        self.event = event
        self.epoch = event.time
        self.origin = event.origin
        self.frame = event.frame

        # Fill in default values for subfields as attributes
        self.dep = Empty.EMPTY
        self.arr = Empty.EMPTY
        self.dep_lt = Empty.EMPTY
        self.arr_lt = Empty.EMPTY
        self.perp = Empty.EMPTY
        self.vflat = Vector3.ZERO

        # Overwrite with given subfields
        self.subfields = {}
        for (key,subfield) in subfields.iteritems():
            self.insert_subfield(key, subfield.as_readonly())

        # Used if needed
        self.filled_ssb = None
        self.filled_shape = None
        self.filled_mask = None

    @property
    def pos(self):
        return self.state.without_derivs()

    @property
    def vel(self):
        if hasattr(self.state, 'd_dt'):
            return self.state.d_dt
        else:
            return Vector3.ZERO

    @property
    def origin_id(self):
        return self.origin.path_id

    @property
    def frame_id(self):
        return self.frame.frame_id

    @property
    def shape(self):
        if self.filled_shape is None:
            self.filled_shape = Qube.broadcasted_shape(self.time,
                                                       self.state,
                                                       self.origin,
                                                       self.frame)
        return self.filled_shape

    @property
    def mask(self):
        if self.filled_mask is None:
            self.filled_mask = self.time.mask | self.state.mask | self.vel.mask

        return self.filled_mask

    def wrt_ssb(self, derivs=True, quick=False):
        """An equivalent Event, relative to the SSB, in the J2000 frame.
        """

        if self.filled_ssb is None:
            rotation = self.frame.wrt(Frame.J2000)
            xform = rotation.transform_at_time(self.epoch, quick=quick)

            state = xform.unrotate(self.state, derivs=True)

            event_ssb = self.event.wrt_ssb(derivs=True, quick=quick)

            result = Event(self.time + event_ssb.time,
                           state + event_ssb.state, Path.SSB, Frame.J2000)

            for (key, subfield) in self.subfields.iteritems():
                result.insert_subfield(key, xform.unrotate(subfield,
                                                           derivs=True))

            self.filled_ssb = result

        if derivs:
            return self.filled_ssb
        else:
            return self.filled_ssb.without_derivs()

    def clone(self):
        """A shallow copy of the Edelta object."""

        result = Event(self.time, self.state, self.event, **self.subfields)

        result.filled_shape = self.filled_shape
        result.filled_mask  = self.filled_mask
        result.filled_ssb   = self.filled_ssb

    def insert_subfield(self, key, value):
        """Insert a given subfield into this Edelta object."""

        self.subfields[key] = value
        self.__dict__[key] = value      # This makes it an attribute as well

        self.filled_ssb = None          # SSB version is now out of date

    @staticmethod
    def sub_events(event, origin, derivs=True, quick=False):
        """The difference of two events as an Edelta object.

        The difference is returned in the frame of the origin."""

        event_ssb = event.wrt_ssb(derivs, quick=quick)
        origin_ssb = origin.wrt_ssb(derivs, quick=quick)

        rotation = origin.frame.wrt(Frame.J2000)
        xform = rotation.transform_at_time(origin.time, quick=quick)

        if derivs:
            state = xform.rotate(event_ssb.state - origin_ssb.state,
                                 derivs=True)
        else:
            state = xform.rotate_pos_vel(event_ssb.pos - origin_ssb.pos,
                                         event_ssb.vel - origin_ssb.vel)

        result = Edelta(event_ssb.time - origin.time, state, origin)

        for key in event_ssb.subfields.keys():
            subfield = event_ssb.subfields[key]
            result.insert_subfield(key, xform.rotate(subfield, derivs))

        return result

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
