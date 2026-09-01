##########################################################################################
# tests/test_event.py
##########################################################################################

import numpy as np
import pytest

from polymath       import Scalar, Vector3
from oops.body      import Body
from oops.event     import Event
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
