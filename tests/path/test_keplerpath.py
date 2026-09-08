##########################################################################################
# tests/path/test_keplerpath.py
##########################################################################################

import numpy as np
import pytest

from polymath       import Scalar, Vector3
from oops.body      import Body
from oops.constants import C
from oops.event     import Event
from oops.frame     import Frame
from oops.gravity   import Gravity
from oops.path      import Path, KeplerPath

# Element indices, as defined by KeplerPath:
#   SEMIM = 0    semimajor axis (km)
#   MEAN0 = 1    mean longitude at epoch (radians)
#   DMEAN = 2    mean motion (radians/s)
#   ECCEN = 3    eccentricity
#   PERI0 = 4    pericenter at epoch (radians)
#   DPERI = 5    pericenter precession rate (radians/s)
#   INCLI = 6    inclination (radians)
#   NODE0 = 7    longitude of ascending node at epoch (radians)
#   DNODE = 8    nodal regression rate (radians/s)
# Each wobble appends a further (amplitude, phase, rate) triple.

_NBASE = 9
_BASE_RATES = (2, 5, 8)         # elements multiplied by time, in radians/s

# A perturbation of a rate element changes a phase by (step * time), so its step is
# bounded by the phase excursion it produces rather than by its own magnitude.
_PHASE_STEP = 1.e-2             # radians
_A = 140000.                    # semimajor axis of the test orbit, km

# Every partial must match to _TOLERANCE, except where the round-off in its own finite
# difference cannot support that; there it is held to _NOISE_MARGIN times the round-off,
# which is a few times the error an exact partial would show. An element whose difference
# clears the round-off by less than _MIN_SIGNAL_TO_NOISE cannot be tested this way at all.
_TOLERANCE = 1.e-4
_NOISE_MARGIN = 50.
_MIN_SIGNAL_TO_NOISE = 1.e3

TIMESTEPS = 100
TIMES = 3600. * np.arange(TIMESTEPS)


def _rate_indices(nwobbles):
    """Indices of the elements expressed as a rate in radians per second.

    Parameters:
        nwobbles (int): Number of wobbles defined for the path.

    Returns:
        set[int]: The element indices that are multiplied by time.
    """

    return set(_BASE_RATES) | {_NBASE + 3*i + 2 for i in range(nwobbles)}


def _positions(kep, times, observer):
    """Positions of the path at the given times.

    Parameters:
        kep (KeplerPath): The path to evaluate.
        times (array): Times in seconds TDB.
        observer (bool): True to return the position relative to the observer in J2000;
            False to return the position in the planet's frame.

    Returns:
        ndarray: An array of shape (len(times), 3), in km.
    """

    if observer:
        return kep.event_at_time(times, partials=False).pos.vals

    return kep._xyz_planet(times, partials=False)[0].vals


def _partials(kep, times, observer):
    """Analytic partial derivatives of position with respect to the elements.

    Parameters:
        kep (KeplerPath): The path to evaluate.
        times (array): Times in seconds TDB.
        observer (bool): True for the position relative to the observer in J2000; False
            for the position in the planet's frame.

    Returns:
        ndarray: An array of shape (len(times), 3, nelements), in km per element unit.
    """

    if observer:
        return kep.event_at_time(times, partials=True).pos.d_delements.vals

    return kep._xyz_planet(times, partials=True)[0].d_delements.vals


def _element_partial_errors(kep, times, observer=False):
    """Relative errors in the analytic partials, one value per element.

    Each analytic partial is compared against a central finite difference. The step for
    each element is chosen so that it moves the body by a fixed distance, which keeps the
    difference well above the round-off floor; for an element expressed as a rate, the
    step is also bounded so that the phase it accumulates stays in the linear regime.

    The partial of an oscillating element passes through zero, so each error is
    normalized by the largest value of that element's partial over the times sampled,
    not by its value at one time.

    Parameters:
        kep (KeplerPath): The path to test.
        times (array): Times in seconds TDB.
        observer (bool, optional): True to test the position relative to the observer in
            J2000; False (the default) to test the position in the planet's frame.

    Returns:
        tuple: (errors, limits, checked), where `errors` holds one relative error per
        element, `limits` holds the error each element could attain if it were exact,
        given the round-off in its own finite difference, and `checked` is True for each
        element whose difference rises far enough above the round-off to mean anything.
    """

    analytic = _partials(kep, times, observer)
    pos = _positions(kep, times, observer)

    # The step is sized to move the body by a fixed fraction of the orbit. Cancellation
    # against the observer distance costs several digits, so that test takes a larger
    # step, trading truncation error for a stronger signal.
    target = (1.e-4 if observer else 1.e-5) * _A
    roundoff = np.abs(pos).max() * 2.2e-16
    tmax = np.abs(times).max()
    rates = _rate_indices(kep._nwobbles)

    tweaked = KeplerPath(kep._planet, kep._epoch, kep._elements.copy(), kep._observer,
                         wobbles=kep._wobbles)
    params = kep.get_elements()

    errors = np.zeros(kep._nelements)
    limits = np.zeros(kep._nelements)
    checked = np.zeros(kep._nelements, dtype='bool')
    for e in range(kep._nelements):
        d_analytic = analytic[..., :, e]
        biggest = np.linalg.norm(d_analytic, axis=-1).max()

        cap = _PHASE_STEP / tmax if e in rates else _PHASE_STEP
        step = min(target / biggest, cap) if biggest > 0. else cap

        hi = params.copy()
        lo = params.copy()
        hi[e] += step
        lo[e] -= step

        tweaked.set_params(hi)
        pos_hi = _positions(tweaked, times, observer)
        tweaked.set_params(lo)
        pos_lo = _positions(tweaked, times, observer)

        d_numeric = (pos_hi - pos_lo) / (2. * step)
        scale = np.linalg.norm(d_numeric, axis=-1).max()

        errors[e] = np.linalg.norm(d_analytic - d_numeric, axis=-1).max() / max(scale,
                                                                               1.e-300)
        # An element whose difference barely clears the round-off in the two positions
        # cannot be validated at all; one that clears it by a wide margin can be held to
        # a correspondingly tighter error.
        signal_to_noise = scale * step / roundoff
        limits[e] = _NOISE_MARGIN / signal_to_noise
        checked[e] = signal_to_noise > _MIN_SIGNAL_TO_NOISE

    return (errors, limits, checked)


def _velocity_errors(kep, times, dt=1.):
    """Relative errors in the analytic velocity, one value per time.

    The velocity returned with the position is compared against a central difference of
    the position in time.

    Parameters:
        kep (KeplerPath): The path to test.
        times (array): Times in seconds TDB.
        dt (float, optional): Time step for the central difference, in seconds; default 1.

    Returns:
        ndarray: The relative error at each time.
    """

    (_, d_pos_dt) = kep._xyz_planet(times, partials=True)

    pos_hi = kep._xyz_planet(times + dt, partials=False)[0].vals
    pos_lo = kep._xyz_planet(times - dt, partials=False)[0].vals
    numeric = (pos_hi - pos_lo) / (2. * dt)

    return (np.linalg.norm(d_pos_dt.vals - numeric, axis=-1)
            / np.linalg.norm(numeric, axis=-1))


def _kepler(extra=(), wobbles=(), **kwargs):
    """A KeplerPath about Saturn using the standard test orbit.

    Parameters:
        extra (tuple, optional): Additional elements appended for the wobbles.
        wobbles (tuple, optional): Names of the wobbles to apply.
        **kwargs: Any further keyword arguments for the KeplerPath constructor.

    Returns:
        KeplerPath: The path, observed from Earth unless an observer is given.
    """

    saturn = Gravity.lookup('SATURN')
    elements = (_A, 1., saturn.n(_A),
                0.2, 3., saturn.dperi_dt(_A),
                0.1, 5., saturn.dnode_dt(_A)) + extra

    kwargs.setdefault('observer', Path.as_path('EARTH'))
    return KeplerPath(Body.lookup('SATURN'), 0., elements, wobbles=wobbles, **kwargs)


def _orbits():
    """The (id, extra elements, wobble names) of each orbit exercised by the tests.

    Returns:
        list[tuple]: One entry per orbit, suitable for parametrization.
    """

    saturn = Gravity.lookup('SATURN')
    n = saturn.n(_A)
    dperi_dt = saturn.dperi_dt(_A)
    dnode_dt = saturn.dnode_dt(_A)

    return [
        ('no wobble', (), ()),
        ('mean+peri+node', (n * 0.10, 2., n / 100.,
                            dperi_dt * 0.08, 4., n / 50.,
                            dnode_dt * 0.12, 6., n / 200.), ('mean', 'peri', 'node')),
        ('a',   (_A * 0.10, 2., n / 100.), ('a',)),
        ('e',   (0.1, 4., n / 50.), ('e',)),
        ('i',   (0.15, 2., n / 150.), ('i',)),
        ('e2d', (1.e-4, 3., dperi_dt / 100.), ('e2d',)),
        ('i2d', (1.e-4, 2., dnode_dt / 150.), ('i2d',)),
        ('i2d+e2d+a', (1.e-4, 2., dperi_dt / 150.,
                       2.e-4, 3., dnode_dt / 200.,
                       _A * 1.e-3, 4., n / 150.), ('i2d', 'e2d', 'a')),
    ]


@pytest.fixture(scope="module", autouse=True)
def _solar_system():
    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system("2000-01-01", "2010-01-01")
    yield
    Frame._reset_caches()
    Path._reset_caches()
    Body.reset_registry()


@pytest.fixture(params=_orbits(), ids=lambda orbit: orbit[0])
def orbit(request):
    """One test orbit, as (extra elements, wobble names)."""

    return request.param[1:]


def test_element_partials_in_planet_frame(orbit) -> None:
    """Each analytic partial matches a central finite difference in the planet's frame."""

    kep = _kepler(*orbit)
    (errors, limits, checked) = _element_partial_errors(kep, TIMES)

    assert np.all(checked), f'element(s) {np.where(~checked)[0]} could not be tested'

    allowed = np.maximum(_TOLERANCE, limits)
    worst = (errors / allowed).argmax()
    assert errors[worst] < allowed[worst], (f'element {worst}: {errors[worst]:.3e} '
                                            f'exceeds {allowed[worst]:.3e}')


def test_element_partials_in_observer_frame(orbit) -> None:
    """Each analytic partial matches a finite difference in the observer's frame.

    Differencing against the distance to the observer costs several digits to
    cancellation, so the elements whose signal falls into the round-off are skipped; the
    planet-frame test covers every element.
    """

    kep = _kepler(*orbit)
    (errors, limits, checked) = _element_partial_errors(kep, TIMES, observer=True)

    # The nine orbital elements are always strong enough to test.
    assert np.all(checked[:_NBASE])

    allowed = np.maximum(_TOLERANCE, limits)
    worst = np.where(checked, errors / allowed, 0.).argmax()
    assert errors[worst] < allowed[worst], (f'element {worst}: {errors[worst]:.3e} '
                                            f'exceeds {allowed[worst]:.3e}')


def test_velocity_matches_position_derivative(orbit) -> None:
    """The velocity returned with the position is its derivative in time."""

    kep = _kepler(*orbit)

    assert _velocity_errors(kep, TIMES).max() < 1.e-6


def test_partials_of_a_wobble_are_nonzero() -> None:
    """A wobble contributes to the partials of the elements it modifies.

    Its own elements move the body, and the partials of the underlying orbital element
    differ from those of the same orbit without the wobble.
    """

    saturn = Gravity.lookup('SATURN')
    n = saturn.n(_A)

    plain = _kepler()
    wobbled = _kepler((0.1, 4., n / 50.), ('e',))

    d_plain = _partials(plain, TIMES, observer=False)
    d_wobbled = _partials(wobbled, TIMES, observer=False)

    # The three elements of the wobble itself move the body.
    for e in range(_NBASE, wobbled._nelements):
        assert np.abs(d_wobbled[..., :, e]).max() > 0.

    # The eccentricity partial is changed by the wobble that modulates it.
    eccen = 3
    difference = np.abs(d_wobbled[..., :, eccen] - d_plain[..., :, eccen]).max()
    assert difference > 0.01 * np.abs(d_plain[..., :, eccen]).max()


def test_photon_to_event_without_observer() -> None:
    """Without an observer, the light time is solved to the body itself."""

    kep = _kepler(observer=None, path_id='kepler_unobserved')

    arrival_time = Scalar(1.e8 + np.arange(5) * 100.)
    arrival = Event(arrival_time, (Vector3.ZERO, Vector3.ZERO), 'EARTH', 'J2000')
    (path_event, _) = kep.photon_to_event(arrival)

    # The ray length and the light travel time agree exactly.
    ratio = (path_event.dep_j2000.norm() / (C * path_event.dep_lt)).vals
    assert np.max(np.abs(ratio - 1.)) < 1.e-12


def test_photon_to_event_with_observer() -> None:
    """With an observer, the light time is solved to the planet rather than the body."""

    kep = _kepler(path_id='kepler_observed')

    arrival_time = Scalar(1.e8 + np.arange(5) * 100.)
    arrival = Event(arrival_time, (Vector3.ZERO, Vector3.ZERO), 'EARTH', 'J2000')
    (path_event, arrival_event) = kep.photon_to_event(arrival)

    # The photon departs before it arrives, so the departure time is measured forward and
    # the arrival time backward, as in Path._solve_photon.
    assert np.all(path_event.time.vals < arrival_time.vals)
    assert np.all(path_event.dep_lt.vals > 0.)
    assert np.all(arrival_event.arr_lt.vals < 0.)
    assert arrival_event.arr_lt == -path_event.dep_lt

    # The ray is the same vector at both ends. Its length matches the light travel time
    # only to the radial part of the orbital offset divided by the range, which is
    # bounded by 1e-4 for this orbit.
    assert arrival_event.arr_j2000 == path_event.dep_j2000
    ratio = (path_event.dep_j2000.norm() / (C * path_event.dep_lt)).vals
    assert np.max(np.abs(ratio - 1.)) < 1.e-4

    # The departure precedes the arrival by exactly the light travel time.
    assert arrival_time - path_event.time == path_event.dep_lt

##########################################################################################
