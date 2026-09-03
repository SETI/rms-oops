##########################################################################################
# tests/test_event.py
##########################################################################################

import numpy as np
import pytest

from polymath       import Scalar, Vector3
from oops.body      import Body
from oops.config    import LOGGING
from oops.event     import Event
from oops.frame     import Frame, Rotation
from oops.path      import Path
from oops.constants import C, RPD


@pytest.fixture(scope='module', autouse=True)
def _solar_system():
    Body._undefine_solar_system()
    Body.define_solar_system('1990-01-01', '2010-01-01')

def test_event():
    import cspyce

    np.random.seed(5531)

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
        assert abs(np.sin(delta[k]) - BETA * np.sin(angles[k])) < 1.e-6

    ######################################################################################
    # Test aberration magnitudes and directions to first order
    ######################################################################################

    BETA = 0.001
    DEL = 3.e-9
    SPEED = BETA * C        # largest speed we care about is 300 km/s
    HALFPI = np.pi/2

    # Incoming aberration in the forward direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
    ev.arr = -Vector3.ZAXIS
    assert Vector3.ZAXIS.sep(ev.neg_arr_ap) == 0.

    # Incoming aberration in the side direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.arr = -Vector3.YAXIS
    assert abs(Vector3.XAXIS.sep(ev.neg_arr_ap) - (HALFPI-BETA)) < DEL

    # Outgoing aberration in the forward direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.dep = Vector3.XAXIS
    assert Vector3.XAXIS.sep(ev.dep_ap) == 0.

    # Incoming aberration in the side direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.dep = Vector3.YAXIS
    assert abs(Vector3.XAXIS.sep(ev.dep_ap) - (HALFPI+BETA)) < DEL

    # Incoming aberration in the forward direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.arr_ap = -Vector3.XAXIS
    assert Vector3.XAXIS.sep(ev.neg_arr_ap) == 0.

    # Incoming aberration in the side direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.arr_ap = -Vector3.YAXIS
    assert abs(Vector3.XAXIS.sep(ev.neg_arr) - (HALFPI+BETA)) < DEL

    # Outgoing aberration in the forward direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.dep = Vector3.XAXIS
    assert Vector3.XAXIS.sep(ev.dep_ap) == 0.

    # Incoming aberration in the side direction
    ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
    ev.dep_ap = Vector3.YAXIS
    assert abs(Vector3.XAXIS.sep(ev.dep) - (HALFPI-BETA)) < DEL

    ######################################################################################
    # Test compatibility with SPICE toolkit and with the exact calculation
    ######################################################################################

    angles = np.arange(181.)
    cspyce_arr_ap = []
    cspyce_dep_ap = []
    for angle in angles:
        vobs = np.array([SPEED, 0., 0.])

        # Note the sign change on pobj, because we consider the photon's
        # direction, not the direction to the target
        pobj = np.array([-np.cos(angle * RPD),
                         -np.sin(angle * RPD), 0.])
        appobj = cspyce.stelab(pobj, vobs)
        cspyce_arr_ap.append(np.arctan2(-appobj[1], -appobj[0]))

        pobj = np.array([np.cos(angle * RPD),
                         np.sin(angle * RPD), 0.])
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
        assert abs(cspyce_arr_ap[k] - exact_arr_ap[k]) < 1.e-6
        assert abs(arr_ap - exact_arr_ap[k]) < 1.e-15

    for k in range(181):
        dep_ap = np.arctan2(ev.dep_ap[k].vals[1], ev.dep_ap[k].vals[0])
        assert abs(cspyce_dep_ap[k] - exact_dep_ap[k]) < 1.e-6
        assert abs(dep_ap - exact_dep_ap[k]) < 1.e-15

    ######################################################################################
    # Test aberration inversions
    ######################################################################################

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

    assert (ev2.arr_ap.unit() - ev1.arr_ap.unit()).norm().max() < 1.e-15

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

    assert (ev2.arr_ap.unit() - ev1.arr_ap.unit()).norm().max() < 1.e-15

    ######################################################################################
    # Subfield checks
    ######################################################################################

    for (origin, frame) in [('SSB', 'J2000'),
                            ('EARTH', 'IAU_EARTH'),
                            ('PLUTO', 'IAU_EARTH')]:

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ##################################################################################
        # Define arr
        ##################################################################################

        ev.arr = (1,2,3)
        assert ev._arr == Vector3((1.,2.,3.))
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        assert ev._neg_arr is None
        assert ev.neg_arr == Vector3((-1.,-2.,-3.))
        assert ev.neg_arr is ev._neg_arr
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        try:
            ev.neg_arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        # Let arr_ap and ssb be filled in
        _ = ev.arr_ap
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define arr_ap
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.arr_ap = (1,2,3)
        assert ev._arr_ap == Vector3((1.,2.,3.))
        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        assert ev.neg_arr_ap == Vector3((-1.,-2.,-3.))
        assert ev._arr_ap == Vector3((1.,2.,3.))
        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._ssb is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._ssb is None

        # Let arr and ssb be filled in
        _ = ev.arr
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 is ev.arr
            assert ev.arr_ap_j2000 is ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define arr_j2000
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.arr_j2000 = (1,2,3)
        assert ev._ssb is not None
        assert ev.ssb._arr == Vector3((1.,2.,3.))
        assert ev.ssb._arr_ap is None
        assert ev.ssb._neg_arr is None
        assert ev.ssb._neg_arr_ap is None

        assert ev._arr is not None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr is not None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        try:
            ev.neg_arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        # Let arr_ap and ssb be filled in
        _ = ev.arr_ap
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define arr_ap_j2000
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.arr_ap_j2000 = (1,2,3)
        assert ev._ssb is not None
        assert ev.ssb._arr_ap == Vector3((1.,2.,3.))
        assert ev.ssb._arr is None
        assert ev.ssb._neg_arr is None
        assert ev.ssb._neg_arr_ap is None

        assert ev._arr_ap is not None
        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is not None
        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        try:
            ev.neg_arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None

        # Let arr and ssb be filled in
        _ = ev.arr_ap
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define neg_arr
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.neg_arr = (-1,-2,-3)
        assert ev._arr == Vector3((1.,2.,3.))
        assert ev._neg_arr == Vector3((-1.,-2.,-3.))
        assert ev._arr_ap is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        try:
            ev.neg_arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr_ap is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        # Let arr_ap and ssb be filled in
        _ = ev.arr_ap
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define neg_arr_ap
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.neg_arr_ap = (-1,-2,-3)
        assert ev._arr_ap == Vector3((1.,2.,3.))
        assert ev._neg_arr_ap == Vector3((-1.,-2.,-3.))
        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._ssb is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._ssb is None

        try:
            ev.neg_arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._arr is None
        assert ev._neg_arr is None
        assert ev._ssb is None

        # Let arr and ssb be filled in
        _ = ev.arr
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define neg_arr_j2000
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.neg_arr_j2000 = (-1,-2,-3)
        assert ev._ssb is not None
        assert ev.ssb._arr == Vector3((1.,2.,3.))
        assert ev.ssb._neg_arr == Vector3((-1.,-2.,-3.))
        assert ev.ssb._arr_ap is None
        assert ev.ssb._neg_arr_ap is None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.neg_arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        # Let arr_ap and ssb be filled in
        _ = ev.arr_ap
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define neg_arr_ap_j2000
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._arr is None
        assert ev._arr_ap is None
        assert ev._neg_arr is None
        assert ev._neg_arr_ap is None
        assert ev._ssb is None

        ev.neg_arr_ap_j2000 = (-1,-2,-3)
        assert ev._ssb is not None
        assert ev.ssb._arr_ap == Vector3((1.,2.,3.))
        assert ev.ssb._neg_arr_ap == Vector3((-1.,-2.,-3.))
        assert ev.ssb._arr_ap is not None
        assert ev.ssb._neg_arr_ap is not None

        try:
            ev.arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.arr_ap = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        try:
            ev.neg_arr = (1,2,3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        # Let arr and ssb be filled in
        _ = ev.arr
        assert (ev.arr_ap - ev.arr).norm() < 5*BETA
        assert ev.neg_arr == -ev.arr
        assert ev.neg_arr_ap == -ev.arr_ap
        assert ev.neg_arr_j2000 == -ev.arr_j2000
        assert ev.neg_arr_ap_j2000 == -ev.arr_ap_j2000

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.arr_j2000 == ev.arr
            assert ev.arr_ap_j2000 == ev.arr_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.arr is not None
            assert ev._ssb.arr_ap is not None

        ##################################################################################
        # Define dep
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._dep is None
        assert ev._dep_ap is None

        ev.dep = (-1,2,-3)
        assert ev._dep == Vector3((-1.,2.,-3.))
        assert ev._dep_ap is None
        assert ev._ssb is None

        try:
            ev.dep_ap = (-1,2,-3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._dep_ap is None
        assert ev._ssb is None

        # Fill in dep_ap and ssb
        _ = ev.dep_ap
        assert (ev.dep_ap - ev.dep).norm() < 5*BETA

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.dep_j2000 == ev.dep
            assert ev.dep_ap_j2000 == ev.dep_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.dep is not None
            assert ev._ssb.dep_ap is not None

        ##################################################################################
        # Define dep_ap
        ##################################################################################

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
        assert ev._dep is None
        assert ev._dep_ap is None

        ev.dep_ap = (-1,2,-3)
        assert ev._dep_ap == Vector3((-1.,2.,-3.))
        assert ev._dep is None
        assert ev._ssb is None

        try:
            ev.dep_ap = (-1,2,-3)
            assert False, 'ValueError not raised'
        except ValueError:
            pass

        assert ev._dep is None
        assert ev._ssb is None

        # Fill in dep and ssb
        _ = ev.dep
        assert (ev.dep_ap - ev.dep).norm() < 5*BETA

        if (origin, frame) == ('SSB', 'J2000'):
            assert ev is ev._ssb
            assert ev.dep_j2000 == ev.dep
            assert ev.dep_ap_j2000 == ev.dep_ap
        else:
            assert ev._ssb is not None
            assert ev._ssb.dep is not None
            assert ev._ssb.dep_ap is not None


def _event_in_a_rotating_frame():
    """An Event on Earth's surface, expressed in a frame that rotates relative to J2000.

    Returns:
        Event: The event, carrying arrival and departure photons.
    """

    event = Event(Scalar(0.), (Vector3((6.4e3, 0., 0.)), Vector3((0., 0.4, 0.))),
                  'EARTH', 'IAU_EARTH')
    event.arr = Vector3((-1., 0., 0.))
    event.arr_lt = Scalar(-0.5)
    event.dep = Vector3((0., -1., 0.))
    event.dep_lt = Scalar(0.7)

    return event


@pytest.mark.parametrize('method, holder, name',
                         [('with_pos_derivs', '_state', 'pos'),
                          ('with_lt_derivs',  'arr_lt',  'lt'),
                          ('with_dep_derivs', 'dep_ap',  'dep'),
                          ('with_dlt_derivs', 'dep_lt',  'dlt')])
def test_with_derivs_inserts_the_derivative(method, holder, name) -> None:
    """Each `with_*_derivs` method returns a clone carrying its unit derivative."""

    event = _event_in_a_rotating_frame()

    result = getattr(event, method)()

    assert name in getattr(result, holder).derivs

    # Asking again returns the same event rather than inserting the derivative twice
    assert getattr(result, method)() is result


def test_with_pos_derivs_gives_the_rotation_into_j2000() -> None:
    """The SSB clone's position derivative is the rotation out of the event's frame.

    The derivative in the event's own frame is the identity, so the derivative of the
    SSB position with respect to it is exactly the transform between the two.
    """

    event = _event_in_a_rotating_frame()
    _ = event.wrt_ssb()                     # so the event has an SSB counterpart

    result = event.with_pos_derivs()

    assert np.allclose(result._state.d_dpos.vals, np.eye(3), atol=1.e-15)
    assert np.allclose(result.ssb._state.d_dpos.vals,
                       result.xform_to_j2000.matrix.vals, atol=1.e-15)


def test_with_pos_derivs_matches_a_numerical_derivative() -> None:
    """The SSB position derivative matches moving the event and re-measuring it."""

    def ssb_position(x):
        event = Event(Scalar(0.), (Vector3((x, 0., 0.)), Vector3((0., 0.4, 0.))),
                      'EARTH', 'IAU_EARTH')
        return event.wrt_ssb().pos.vals

    event = _event_in_a_rotating_frame()
    _ = event.wrt_ssb()
    analytic = event.with_pos_derivs().ssb._state.d_dpos.vals[..., 0]

    # A large step, because the SSB position is ~1e8 km and the difference is not
    step = 100.
    numeric = (ssb_position(6.4e3 + step) - ssb_position(6.4e3 - step)) / (2. * step)

    assert np.allclose(numeric, analytic, rtol=1.e-9)


def test_reading_vflat_does_not_block_assigning_it() -> None:
    """The zero returned for an undefined surface velocity is not saved as a value."""

    event = Event(Scalar(0.), (Vector3.ZERO, Vector3.ZERO), 'SSB', 'J2000')

    assert event.vflat == Vector3.ZERO       # the default, which must not be recorded

    event.vflat = Vector3((1., 0., 0.))
    assert event.vflat == Vector3((1., 0., 0.))


def test_vflat_can_only_be_assigned_once() -> None:
    """An explicit surface velocity is still refused a second value."""

    event = Event(Scalar(0.), (Vector3.ZERO, Vector3.ZERO), 'SSB', 'J2000')
    event.vflat = Vector3((1., 0., 0.))

    with pytest.raises(ValueError, match='already defined'):
        event.vflat = Vector3((2., 0., 0.))


##########################################################################################
# Event properties, copies, masking, and geometry
##########################################################################################

def _simple_event() -> Event:
    """A shapeless Event at the solar system barycenter."""

    return Event(Scalar(0.), Vector3((1.e5, 2.e5, 3.e5)), 'SSB', 'J2000')


def _sequence_event() -> Event:
    """A four-element Event, so masking and shrinking have something to act on."""

    return Event(Scalar([0., 1., 2., 3.]),
                 Vector3([(1., 0., 0.), (2., 0., 0.), (3., 0., 0.), (4., 0., 0.)]),
                 'SSB', 'J2000')


def test_event_reports_its_origin_and_frame_ids() -> None:
    """The IDs name the registered origin path and coordinate frame."""

    event = _simple_event()

    assert event.origin_id == 'SSB'
    assert event.frame_id == 'J2000'


def test_event_shape_and_size() -> None:
    """A shapeless Event holds a single value."""

    event = _simple_event()

    assert event.shape == ()
    assert event.size == 1


def test_event_sequence_shape_and_size() -> None:
    """An Event broadcasts its properties to a common shape."""

    event = _sequence_event()

    assert event.shape == (4,)
    assert event.size == 4


def test_event_mask_and_antimask_are_complementary() -> None:
    """An unmasked Event is entirely unmasked."""

    event = _simple_event()

    assert event.mask == False           # noqa: E712  (a numpy bool, not the singleton)
    assert event.antimask == True        # noqa: E712


def test_event_has_no_photons_until_they_are_assigned() -> None:
    """An Event carries neither arrivals nor departures until one is set."""

    event = _simple_event()

    assert not event.has_arrivals()
    assert not event.has_departures()


def test_event_records_an_arriving_photon() -> None:
    """Assigning `arr` gives the Event an arrival direction."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))

    assert event.has_arrivals()
    assert event.arr == Vector3((0., 0., -1.))


def test_event_records_a_departing_photon() -> None:
    """Assigning `dep` gives the Event a departure direction."""

    event = _simple_event()
    event.dep = Vector3((0., 0., 1.))

    assert event.has_departures()
    assert event.dep == Vector3((0., 0., 1.))


def test_event_neg_arr_is_the_reversed_arrival() -> None:
    """`neg_arr` is the negative of `arr`, which is needed often enough to name."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))

    assert event.neg_arr == -event.arr


def test_event_vflat_defaults_to_zero() -> None:
    """The surface velocity is zero if it was never defined."""

    assert _simple_event().vflat == Vector3.ZERO


def test_event_perp_is_none_until_assigned() -> None:
    """The normal vector is undefined until the Event is placed on a surface."""

    assert _simple_event().perp is None


def test_event_subfields_become_attributes() -> None:
    """An inserted subfield is readable by name and through get_subfield."""

    event = _simple_event()
    event.insert_subfield('sample', 42)

    assert event.sample == 42
    assert event.get_subfield('sample') == 42
    assert event.subfields['sample'] == 42


def test_event_str_names_its_origin_and_frame() -> None:
    """The string form identifies the time, position, shape, origin, and frame."""

    text = str(_simple_event())

    assert 'SSB' in text
    assert 'J2000' in text


def test_event_copy_keeps_the_photons() -> None:
    """A shallow copy carries the arrival and departure vectors over."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))
    copied = event.copy()

    assert copied.has_arrivals()
    assert copied.arr == event.arr


def test_event_copy_can_omit_the_arrivals() -> None:
    """"arr" omits all of the arrival vectors from the copy."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))
    copied = event.copy(omit=('arr',))

    assert not copied.has_arrivals()


def test_event_copy_can_omit_the_departures() -> None:
    """"dep" omits all of the departure vectors from the copy."""

    event = _simple_event()
    event.dep = Vector3((0., 0., 1.))
    copied = event.copy(omit=('dep',))

    assert not copied.has_departures()


def test_event_as_all_masked() -> None:
    """The result is a copy of the Event with everything masked."""

    masked = _sequence_event().as_all_masked()

    assert np.all(masked.mask)
    assert masked.shape == (4,)


def test_event_as_all_masked_can_broadcast() -> None:
    """A broadcast shape enlarges the masked result."""

    masked = _simple_event().as_all_masked(broadcast=(2, 3))

    assert masked.shape == (2, 3)
    assert np.all(masked.mask)


def test_event_as_all_masked_can_change_the_frame() -> None:
    """A given origin and frame replace those of this Event."""

    masked = _simple_event().as_all_masked(origin=Path.SSB, frame=Frame.J2000)

    assert masked.origin_id == 'SSB'
    assert masked.frame_id == 'J2000'


def test_event_mask_where_applies_a_new_mask() -> None:
    """mask_where masks the elements the argument selects."""

    masked = _sequence_event().mask_where(np.array([True, False, False, False]))

    assert list(masked.mask) == [True, False, False, False]


def test_event_remask_replaces_the_mask() -> None:
    """remask installs the given mask in place of the existing one."""

    masked = _sequence_event().remask(np.array([True, False, False, False]))

    assert list(masked.mask) == [True, False, False, False]


def test_event_shrink_keeps_only_the_selected_elements() -> None:
    """A shrunken Event holds only the values where the antimask is True."""

    antimask = np.array([True, False, True, False])
    shrunk = _sequence_event().shrink(antimask)

    assert shrunk.shape == (2,)
    assert shrunk.time == Scalar([0., 2.])


def test_event_unshrink_restores_the_original_shape() -> None:
    """Expanding a shrunken Event masks the values the antimask discarded."""

    antimask = np.array([True, False, True, False])
    event = _sequence_event()
    restored = event.shrink(antimask).unshrink(antimask)

    assert restored.shape == (4,)
    assert list(restored.mask) == [False, True, False, True]
    assert restored.time[0] == Scalar(0.)
    assert restored.time[2] == Scalar(2.)


@pytest.mark.parametrize('antimask', [None, True], ids=['None', 'True'])
def test_event_shrink_keeping_everything_returns_the_same_event(antimask) -> None:
    """A single True, or None, keeps everything, so the Event is returned unchanged."""

    event = _sequence_event()

    assert event.shrink(antimask) is event


def test_event_shrink_ignoring_everything_gives_a_masked_event() -> None:
    """A single False ignores everything, leaving a shapeless masked Event."""

    shrunk = _sequence_event().shrink(False)

    assert shrunk.shape == ()
    assert shrunk.mask == True           # noqa: E712


def test_event_wod_is_cached() -> None:
    """The derivative-free version is cached, unlike without_derivs()."""

    event = _sequence_event()

    assert event.wod is event.wod


def test_event_without_derivs_is_not_cached() -> None:
    """without_derivs() builds a new Event on every call."""

    event = _sequence_event()

    assert event.without_derivs() is not event.without_derivs()


def test_event_collapse_time_leaves_a_long_span_alone() -> None:
    """An Event spanning more than the threshold is returned unchanged."""

    event = Event(Scalar([0., 100.]), Vector3((1., 2., 3.)), 'SSB', 'J2000')

    assert event.collapse_time() is event


def test_event_collapse_time_leaves_a_shapeless_event_alone() -> None:
    """An Event with a single time has no span to collapse."""

    event = _simple_event()

    assert event.collapse_time() is event


def test_event_collapse_time_is_disabled_by_a_zero_threshold() -> None:
    """A threshold of zero collapses nothing."""

    event = Event(Scalar([0., 100.]), Vector3((1., 2., 3.)), 'SSB', 'J2000')

    assert event.collapse_time(threshold=0.) is event


def test_event_collapse_time_replaces_a_short_span_with_the_midtime() -> None:
    """An Event spanning less than the threshold gets a single midtime."""

    event = Event(Scalar([0., 1.e-9]), Vector3((1., 2., 3.)), 'SSB', 'J2000')
    collapsed = event.collapse_time()

    assert collapsed.time == Scalar(0.5e-9)


##########################################################################################
# Photon geometry
##########################################################################################

def _surface_event(arr: Vector3, dep: Vector3) -> Event:
    """An Event on a surface whose normal is the +Z axis, carrying both photons."""

    event = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')
    event.perp = Vector3((0., 0., 1.))
    event.arr = arr
    event.dep = dep

    return event


def test_incidence_angle_is_zero_overhead() -> None:
    """A photon arriving straight down has zero incidence angle."""

    event = _surface_event(Vector3((0., 0., -1.)), Vector3((0., 0., 1.)))

    assert event.incidence_angle().vals == pytest.approx(0., abs=1.e-12)


def test_incidence_angle_is_measured_from_the_normal() -> None:
    """The incidence angle lies between the normal and the reversed arrival."""

    event = _surface_event(Vector3((1., 0., -1.)), Vector3((0., 0., 1.)))

    assert event.incidence_angle().vals == pytest.approx(45. * RPD)


def test_emission_angle_is_zero_overhead() -> None:
    """A photon departing straight up has zero emission angle."""

    event = _surface_event(Vector3((0., 0., -1.)), Vector3((0., 0., 1.)))

    assert event.emission_angle().vals == pytest.approx(0., abs=1.e-12)


def test_emission_angle_is_measured_from_the_normal() -> None:
    """The emission angle lies between the normal and the departing photon."""

    event = _surface_event(Vector3((0., 0., -1.)), Vector3((1., 0., 1.)))

    assert event.emission_angle().vals == pytest.approx(45. * RPD)


def test_phase_angle_is_zero_for_backscatter() -> None:
    """A photon departing back the way it arrived has zero phase angle."""

    event = _surface_event(Vector3((0., 0., -1.)), Vector3((0., 0., 1.)))

    assert event.phase_angle().vals == pytest.approx(0., abs=1.e-12)


def test_phase_angle_separates_the_two_photons() -> None:
    """The phase angle lies between the arrival and the reversed departure."""

    event = _surface_event(Vector3((1., 0., -1.)), Vector3((0., 0., 1.)))

    assert event.phase_angle().vals == pytest.approx(45. * RPD)


def test_ra_and_dec_of_an_arriving_photon() -> None:
    """An arriving direction is reversed, so a photon from +X has RA 0 and dec 0."""

    event = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')
    event.arr = Vector3((-1., 0., 0.))
    (ra, dec) = event.ra_and_dec()

    assert ra.vals == pytest.approx(0., abs=1.e-12)
    assert dec.vals == pytest.approx(0., abs=1.e-12)


def test_ra_and_dec_of_a_photon_from_the_north_pole() -> None:
    """A photon arriving from +Z has declination +90 degrees."""

    event = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')
    event.arr = Vector3((0., 0., -1.))
    (_, dec) = event.ra_and_dec()

    assert dec.vals == pytest.approx(90. * RPD)


def test_ra_and_dec_can_use_the_departing_photon() -> None:
    """The "dep" subfield gives the direction of the departing photon, unreversed."""

    event = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')
    event.dep = Vector3((1., 0., 0.))
    (ra, dec) = event.ra_and_dec(subfield='dep')

    assert ra.vals == pytest.approx(0., abs=1.e-12)
    assert dec.vals == pytest.approx(0., abs=1.e-12)


def test_ra_wraps_into_zero_to_two_pi() -> None:
    """Right ascension is reported in the range 0 to 2*pi."""

    event = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')
    event.arr = Vector3((0., 1., 0.))
    (ra, _) = event.ra_and_dec()

    assert ra.vals == pytest.approx(270. * RPD)


##########################################################################################
# Apparent versus actual photon directions
##########################################################################################

def test_actual_arr_is_the_assigned_direction() -> None:
    """The actual direction is the one that was assigned, without aberration."""

    event = _event_in_a_rotating_frame()

    assert event.actual_arr() == Vector3((-1., 0., 0.))


def test_apparent_arr_differs_from_the_actual_direction() -> None:
    """Stellar aberration tilts the apparent direction away from the actual one."""

    event = _event_in_a_rotating_frame()

    assert event.apparent_arr() != event.actual_arr()


def test_the_aberration_of_the_arrival_is_small() -> None:
    """The observer's motion is far below the speed of light, so the tilt is tiny."""

    event = _event_in_a_rotating_frame()

    assert event.apparent_arr().sep(event.actual_arr()).vals < 1.e-3


def test_actual_dep_is_the_assigned_direction() -> None:
    """The actual direction is the one that was assigned, without aberration."""

    event = _event_in_a_rotating_frame()

    assert event.actual_dep() == Vector3((0., -1., 0.))


def test_apparent_dep_differs_from_the_actual_direction() -> None:
    """Stellar aberration tilts the apparent direction away from the actual one."""

    event = _event_in_a_rotating_frame()

    assert event.apparent_dep() != event.actual_dep()


@pytest.mark.parametrize('method', ['apparent_arr', 'actual_arr',
                                    'apparent_dep', 'actual_dep'])
def test_the_photon_directions_are_cached(method: str) -> None:
    """Each direction is computed once and then kept."""

    event = _event_in_a_rotating_frame()

    assert getattr(event, method)() is getattr(event, method)()


def test_actual_ray_ssb_inverts_apparent_ray_ssb() -> None:
    """The two aberration corrections are inverses of one another."""

    event = _event_in_a_rotating_frame()
    ray = Vector3((1., 0., 0.))

    restored = event.actual_ray_ssb(event.apparent_ray_ssb(ray))

    assert restored.sep(ray).vals == pytest.approx(0., abs=1.e-9)


def test_apparent_ray_ssb_is_not_cached() -> None:
    """The correction depends on the ray given, so there is nothing to cache."""

    event = _event_in_a_rotating_frame()
    ray = Vector3((1., 0., 0.))

    assert event.apparent_ray_ssb(ray) is not event.apparent_ray_ssb(ray)


@pytest.mark.parametrize('method', ['incidence_angle', 'emission_angle', 'phase_angle'])
def test_the_apparent_angles_differ_from_the_actual_ones(method: str) -> None:
    """Accounting for aberration shifts each angle slightly."""

    event = _event_in_a_rotating_frame()
    event.perp = Vector3((1., 0., 0.))

    actual = getattr(event, method)(apparent=False)
    apparent = getattr(event, method)(apparent=True)

    assert actual.vals != apparent.vals
    assert abs(actual.vals - apparent.vals) < 1.e-3


def test_the_angles_are_actual_by_default() -> None:
    """apparent defaults to False on an Event, unlike on a Backplane."""

    event = _event_in_a_rotating_frame()
    event.perp = Vector3((1., 0., 0.))

    assert event.incidence_angle() == event.incidence_angle(apparent=False)


def test_ra_and_dec_in_the_frame_of_the_event() -> None:
    """frame=None reports the direction in this event's own frame."""

    event = _event_in_a_rotating_frame()
    (ra, dec) = event.ra_and_dec(frame=None)

    assert ra.vals == pytest.approx(0., abs=1.e-12)
    assert dec.vals == pytest.approx(0., abs=1.e-12)


def test_ra_and_dec_in_j2000_differs_from_the_event_frame() -> None:
    """The default J2000 frame is rotated relative to the event's own."""

    event = _event_in_a_rotating_frame()

    assert event.ra_and_dec()[0].vals != event.ra_and_dec(frame=None)[0].vals


def test_apparent_ra_and_dec_differ_from_the_geometric_values() -> None:
    """Including stellar aberration moves the apparent position of a star."""

    event = _event_in_a_rotating_frame()

    assert event.ra_and_dec(apparent=True)[0].vals \
           != event.ra_and_dec(apparent=False)[0].vals


##########################################################################################
# Subtracting one event from another
##########################################################################################

def test_sub_gives_the_relative_position() -> None:
    """The difference is this event's position relative to the reference."""

    event = _event_in_a_rotating_frame()
    reference = Event(Scalar(0.), Vector3((0., 0., 0.)), 'EARTH', 'IAU_EARTH')

    difference = event.sub(reference)

    assert difference.pos.vals[0] == pytest.approx(6.4e3)


def test_sub_gives_the_relative_time() -> None:
    """Times in the difference are measured relative to the reference event."""

    event = _event_in_a_rotating_frame()
    reference = Event(Scalar(0.), Vector3((0., 0., 0.)), 'EARTH', 'IAU_EARTH')

    assert event.sub(reference).time == Scalar(0.)


def test_sub_of_an_event_from_itself_is_zero() -> None:
    """An event subtracted from itself leaves nothing."""

    event = _event_in_a_rotating_frame()
    difference = event.sub(event)

    assert difference.pos.vals == pytest.approx([0., 0., 0.], abs=1.e-6)
    assert difference.time == Scalar(0.)


def test_sub_keeps_both_source_events() -> None:
    """The result points back at the two events it was built from."""

    event = _event_in_a_rotating_frame()
    reference = Event(Scalar(0.), Vector3((0., 0., 0.)), 'EARTH', 'IAU_EARTH')
    difference = event.sub(reference)

    assert difference.event is event
    assert difference.reference is reference

##########################################################################################
# Construction, the setters that refuse to be called twice, and the description
##########################################################################################

def test_the_frame_defaults_to_the_frame_of_the_origin() -> None:
    """With no frame given, the event adopts the frame in which its origin is defined."""

    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB')

    assert event.frame is Path.as_waypoint('SSB').frame


def test_the_velocity_of_a_position_without_derivatives_is_zero() -> None:
    """A state given as a position alone is at rest."""

    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', 'J2000')

    assert event.vel == Vector3.ZERO


@pytest.mark.parametrize('name, value',
                         [('arr', Vector3((0., 0., -1.))),
                          ('arr_lt', Scalar(-1.)),
                          ('dep', Vector3((0., 0., 1.))),
                          ('dep_lt', Scalar(1.)),
                          ('perp', Vector3((0., 0., 1.)))])
def test_a_photon_attribute_can_only_be_assigned_once(name: str, value) -> None:
    """An event's photons are set once; a second assignment would be ambiguous."""

    event = _simple_event()
    setattr(event, name, value)

    with pytest.raises(ValueError, match='already defined'):
        setattr(event, name, value)


def test_assigning_a_j2000_departure_rotates_it_into_the_event_frame() -> None:
    """A departure given in J2000 is rotated into the frame of the event."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_ROTATED')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)

    event.dep_ap_j2000 = Vector3((0., 0., 1.))

    assert event.dep_ap_j2000 == Vector3((0., 0., 1.))
    assert event.ssb.dep_ap == Vector3((0., 0., 1.))


def test_assigning_a_j2000_departure_to_an_ssb_event_is_direct() -> None:
    """An event already in SSB/J2000 has nothing to rotate."""

    event = _simple_event()

    event.dep_ap_j2000 = Vector3((0., 0., 1.))

    assert event.dep_ap == Vector3((0., 0., 1.))


def test_assigning_a_perpendicular_fills_in_the_ssb_version() -> None:
    """The barycentric copy of an event picks up the rotated normal."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_PERP')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    _ = event.ssb                                   # build the barycentric copy first

    event.perp = Vector3((0., 0., 1.))

    assert event.ssb.perp == Vector3((0., 0., 1.))  # the z-axis is the rotation axis


def test_assigning_a_flat_velocity_fills_in_the_ssb_version() -> None:
    """The barycentric copy of an event picks up the rotated surface velocity."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_VFLAT')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    _ = event.ssb

    event.vflat = Vector3((0., 0., 1.))

    assert event.ssb.vflat == Vector3((0., 0., 1.))


@pytest.mark.parametrize('size, expected',
                         [(1, 'Scalar(0.0)'), (2, 'Scalar(0. 1.)'),
                          (4, 'Scalar(0.0), ..., Scalar(3.0)')])
def test_the_description_abbreviates_a_long_array(size: int, expected: str) -> None:
    """One value is printed alone, two together, and more as the first and the last."""

    times = np.arange(float(size))
    event = Event(Scalar(times), Vector3(np.zeros((size, 3))), 'SSB', 'J2000')

    assert expected in str(event)


def test_the_description_lists_the_subfields() -> None:
    """Every subfield is named after the shape, origin and frame."""

    event = _simple_event()
    event.insert_subfield('coord1', Scalar(1.))

    assert '; coord1' in str(event)


##########################################################################################
# Copies, masking and derivatives
##########################################################################################

def test_copy_can_omit_a_subfield_that_is_not_there() -> None:
    """Omitting a name the event does not carry is harmless."""

    event = _simple_event()

    assert event.copy(omit=('not_a_subfield',)).pos == event.pos


def test_copy_omits_a_subfield_from_the_barycentric_copy_too() -> None:
    """A subfield dropped from the event is dropped from its SSB counterpart."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_OMIT')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    event.insert_subfield('coord1', Scalar(1.))
    _ = event.ssb

    copied = event.copy(omit=('coord1',))

    assert 'coord1' not in copied.subfields
    assert 'coord1' not in copied.ssb.subfields


def test_mask_where_broadcasts_a_shapeless_subfield() -> None:
    """A subfield that does not yet have the event's shape is broadcast before masking."""

    event = _sequence_event()
    event.insert_subfield('coord1', Scalar(1.))

    masked = event.mask_where(np.array([False, True, False, True]))

    assert list(masked.coord1.mask) == [False, True, False, True]


def test_with_time_derivs_returns_the_same_event_when_it_already_has_them() -> None:
    """The derivative is inserted once; a second request changes nothing."""

    event = _simple_event().with_time_derivs()

    assert event.with_time_derivs() is event


def test_with_time_derivs_inserts_the_derivative_into_the_ssb_copy() -> None:
    """The barycentric copy carries the same time derivative."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_TIME_DERIVS')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    _ = event.ssb

    with_derivs = event.with_time_derivs()

    assert 't' in with_derivs.time.derivs
    assert 't' in with_derivs.ssb.time.derivs


def test_with_los_derivs_returns_the_same_event_when_it_already_has_them() -> None:
    """The derivative is inserted once; a second request changes nothing."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))
    with_derivs = event.with_los_derivs()

    assert with_derivs.with_los_derivs() is with_derivs


##########################################################################################
# Frames, paths, and the errors the geometry raises
##########################################################################################

def test_from_ssb_requires_a_barycentric_event() -> None:
    """An event that is not in SSB/J2000 has nothing to convert from."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_FROM_SSB')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)

    with pytest.raises(ValueError, match='requires a SSB/J2000 event'):
        event.from_ssb('SSB', 'J2000')


def test_wrt_defaults_to_the_origin_and_frame_of_the_event() -> None:
    """With neither given, the event is returned relative to what it already uses."""

    event = _simple_event()

    assert event.wrt().pos == event.pos


def test_wrt_path_of_none_keeps_the_origin_of_the_event() -> None:
    """A path of None means the event keeps its own origin."""

    event = _simple_event()

    assert event.wrt_path(None).pos == event.pos


def test_wrt_frame_of_none_keeps_the_frame_of_the_event() -> None:
    """A frame of None means the event keeps its own frame."""

    event = _simple_event()

    assert event.wrt_frame(None).pos == event.pos


def test_unrotate_by_frame_returns_the_event_to_the_reference_frame() -> None:
    """The event is expressed in the frame the rotation is defined relative to."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_UNROTATE')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    event.arr = Vector3((0., 0., -1.))
    event.insert_subfield('normal', Vector3((0., 0., 1.)))

    unrotated = event.unrotate_by_frame(frame)

    assert unrotated.frame is Frame.J2000.wayframe
    assert unrotated.pos.vals == pytest.approx(event.wrt_frame(Frame.J2000).pos.vals,
                                               abs=1.e-9)

    # The z-axis is the rotation axis, so a normal along it is unchanged
    assert unrotated.normal.vals == pytest.approx((0., 0., 1.), abs=1.e-12)


def test_unrotate_by_frame_can_drop_the_derivatives() -> None:
    """Without derivs, the event is stripped before it is unrotated."""

    frame = Rotation(0.5, 2, Frame.J2000, frame_id='TEST_EVENT_UNROTATE_WOD')
    event = Event(Scalar(0.), Vector3((1.e5, 0., 0.)), 'SSB', frame)
    event.insert_subfield('coord1', Scalar(1.))
    event.coord1.insert_deriv('x', Scalar(1.))

    assert 'x' not in event.unrotate_by_frame(frame, derivs=False).coord1.derivs
    assert 'x' in event.unrotate_by_frame(frame, derivs=True).coord1.derivs


def test_collapse_time_leaves_an_event_with_time_derivatives_alone() -> None:
    """A time derivative names each element, so the times cannot be collapsed."""

    event = _sequence_event().with_time_derivs()

    assert event.collapse_time() is event


def test_collapsing_the_time_is_reported_as_a_diagnostic(
        capsys: pytest.CaptureFixture[str]) -> None:
    """The span that was collapsed is logged when diagnostics are on."""

    event = Event(Scalar([0., 1.e-9]), Vector3([(1., 0., 0.), (2., 0., 0.)]),
                  'SSB', 'J2000')

    LOGGING.on()
    LOGGING.event_time_collapse = True
    try:
        event.collapse_time()
    finally:
        LOGGING.event_time_collapse = False
        LOGGING.off()

    assert 'Event.collapse_time()' in capsys.readouterr().out


def test_subtraction_carries_the_subfields_of_the_barycentric_event() -> None:
    """A vector subfield of the event survives into the difference, unrotated."""

    first = _simple_event()
    first.insert_subfield('normal', Vector3((0., 0., 1.)))
    second = Event(Scalar(0.), Vector3((0., 0., 0.)), 'SSB', 'J2000')

    diff = first.sub(second)

    assert diff.normal == Vector3((0., 0., 1.))


@pytest.mark.parametrize('method, missing',
                         [('incidence_angle', 'arr'),
                          ('emission_angle', 'dep')])
def test_an_angle_needs_the_photon_it_is_measured_from(method: str,
                                                       missing: str) -> None:
    """The incidence angle needs an arrival and the emission angle a departure."""

    event = _simple_event()
    event.perp = Vector3((0., 0., 1.))

    with pytest.raises(ValueError, match=f'undefined {missing}'):
        getattr(event, method)()


@pytest.mark.parametrize('method', ['incidence_angle', 'emission_angle'])
def test_an_angle_needs_a_surface_normal(method: str) -> None:
    """Both angles are measured from the normal, so the normal has to be there."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))
    event.dep = Vector3((0., 0., 1.))

    with pytest.raises(ValueError, match='undefined perpendicular'):
        getattr(event, method)()


@pytest.mark.parametrize('missing', ['arr', 'dep'])
def test_the_phase_angle_needs_both_photons(missing: str) -> None:
    """The phase angle separates the arriving photon from the departing one."""

    event = _simple_event()
    if missing == 'arr':
        event.dep = Vector3((0., 0., 1.))
    else:
        event.arr = Vector3((0., 0., -1.))

    with pytest.raises(ValueError, match=f'undefined {missing}'):
        event.phase_angle()


def test_ra_and_dec_rejects_an_unknown_subfield() -> None:
    """The direction comes from the arriving photon or the departing one."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))

    with pytest.raises(ValueError, match='invalid input value for subfield'):
        event.ra_and_dec(subfield='perp')


def test_ra_and_dec_needs_the_photon_it_is_asked_for() -> None:
    """An event with no departing photon has no direction to report."""

    event = _simple_event()
    event.arr = Vector3((0., 0., -1.))

    with pytest.raises(ValueError, match='undefined light ray'):
        event.ra_and_dec(subfield='dep')

##########################################################################################
