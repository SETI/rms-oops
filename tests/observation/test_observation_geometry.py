##########################################################################################
# tests/observation/test_observation_geometry.py: Observation methods that need real
# SPICE geometry
##########################################################################################

from collections.abc import Iterator
from pathlib import Path as FilePath
from typing import cast

import cspyce
import numpy as np
import pytest

from polymath                          import Scalar, Vector, Vector3
from oops.backplane                    import Backplane
from oops.body                         import Body
from oops.cadence                     import Metronome
from oops.config                       import LOGGING
from oops.event                        import Event
from oops.fov                          import FlatFOV
from oops.frame                        import Frame, TwoVectorFrame
from oops.observation                  import Observation, Snapshot, TimedImage
from oops.path                         import Path
from programs.gold_master.test_support  import TEST_SPICE_PREFIX

from tests.conftest import CORE_KERNELS

# A time within the interval the test kernels cover, and a synthetic camera pointed
# straight at Saturn from Earth. Saturn subtends about 9.1e-5 radians here, so this
# pixel size puts roughly 20 pixels across the disk of a 40x40 image.
TIME = 1.e8
TEXP = 10.
PIXEL = 9.136e-05 / 20.
SHAPE = (40, 40)


@pytest.fixture(scope='module')
def solar_system() -> Iterator[None]:
    """The bodies of the solar system over the interval the test kernels cover."""

    for path in cast(list[FilePath], TEST_SPICE_PREFIX.retrieve(CORE_KERNELS)):
        cspyce.furnsh(path)

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()
    Body._undefine_solar_system()
    Body.define_solar_system('2000-01-01', '2010-01-01')

    yield

    Path._reset_caches()
    Frame._reset_caches()
    Body.reset_registry()


@pytest.fixture(scope='module')
def obs(solar_system: None) -> Snapshot:
    """A Snapshot of Saturn taken from Earth, pointed at the center of the planet."""

    earth = Path.as_path('EARTH')
    saturn = Body.lookup('SATURN')
    los = saturn.path.event_at_time(Scalar(TIME)).wrt_path(earth).pos.unit()

    TwoVectorFrame(Frame.J2000, los, 'z', Vector3.XAXIS, 'x',
                   frame_id='TEST_GEOMETRY_CAMERA')

    return Snapshot(('u', 'v'), TIME, TEXP, FlatFOV((PIXEL, PIXEL), SHAPE),
                    'EARTH', 'TEST_GEOMETRY_CAMERA')


##########################################################################################
# event_at_grid and gridless_event
##########################################################################################

def test_event_at_grid_has_the_shape_of_the_meshgrid(obs: Snapshot) -> None:
    """One arrival event is generated for every sample of the meshgrid."""

    event = obs.event_at_grid(obs.meshgrid())

    assert isinstance(event, Event)
    assert event.shape == SHAPE


def test_event_at_grid_carries_the_arrival_directions(obs: Snapshot) -> None:
    """The event describes photons arriving from the directions of the meshgrid."""

    assert obs.event_at_grid(obs.meshgrid()).has_arrivals()


def test_event_at_grid_defaults_to_the_midtime(obs: Snapshot) -> None:
    """tfrac defaults to 0.5, the middle of the exposure."""

    default = obs.event_at_grid(obs.meshgrid())
    explicit = obs.event_at_grid(obs.meshgrid(), tfrac=0.5)

    assert default.time == explicit.time


@pytest.mark.parametrize('tfrac, expected', [(0., TIME), (1., TIME + TEXP)])
def test_event_at_grid_honors_tfrac(tfrac: float, expected: float,
                                    obs: Snapshot) -> None:
    """tfrac=0 is the beginning of the exposure and 1 is the end."""

    event = obs.event_at_grid(obs.meshgrid(), tfrac=tfrac)

    assert event.time == Scalar(expected)


def test_event_at_grid_accepts_an_absolute_time(obs: Snapshot) -> None:
    """A time replaces the fractional time within the exposure."""

    event = obs.event_at_grid(obs.meshgrid(), tfrac=None, time=Scalar(TIME + 3.))

    assert event.time == Scalar(TIME + 3.)


def test_an_absolute_time_takes_precedence_over_tfrac(obs: Snapshot) -> None:
    """When both are given, the absolute time is the one that is used."""

    event = obs.event_at_grid(obs.meshgrid(), tfrac=0.5, time=Scalar(TIME))

    assert event.time == Scalar(TIME)


def test_gridless_event_has_no_directions(obs: Snapshot) -> None:
    """A gridless event describes the observer without a line of sight."""

    event = obs.gridless_event(obs.meshgrid())

    assert not event.has_arrivals()


def test_gridless_event_is_shapeless_for_an_untimed_observation(obs: Snapshot) -> None:
    """A Snapshot has one time, so its gridless event holds one value."""

    assert obs.gridless_event(obs.meshgrid()).shape == ()


def test_gridless_event_can_be_forced_shapeless(obs: Snapshot) -> None:
    """shapeless=True collapses the event onto the mean of all the times."""

    assert obs.gridless_event(obs.meshgrid(), shapeless=True).shape == ()


def test_gridless_event_accepts_an_absolute_time(obs: Snapshot) -> None:
    """A time replaces the fractional time within the exposure."""

    event = obs.gridless_event(obs.meshgrid(), time=Scalar(TIME + 3.))

    assert event.time == Scalar(TIME + 3.)


def test_gridless_event_sits_on_the_observer_path(obs: Snapshot) -> None:
    """The event is co-located with the instrument."""

    assert obs.gridless_event(obs.meshgrid()).origin == obs.path


##########################################################################################
# scalar_from_indices
##########################################################################################

def test_scalar_from_indices_selects_one_axis() -> None:
    """The named axis is extracted from a Vector of indices."""

    indices = Vector([[1., 2.]])

    assert Observation.scalar_from_indices(indices, 0) == Scalar([1.])
    assert Observation.scalar_from_indices(indices, 1) == Scalar([2.])


def test_scalar_from_indices_returns_none_for_a_missing_axis() -> None:
    """A negative axis is not associated with an array index."""

    assert Observation.scalar_from_indices(Vector([[1., 2.]]), -1) is None


def test_scalar_from_indices_accepts_a_bare_number() -> None:
    """A single number is the index along axis zero."""

    assert Observation.scalar_from_indices(3., 0) == Scalar(3.)


def test_scalar_from_indices_rejects_a_number_on_another_axis() -> None:
    """A single number has no axis but the first."""

    with pytest.raises(IndexError):
        Observation.scalar_from_indices(3., 1)


##########################################################################################
# uv_from_ra_and_dec, uv_from_path, uv_from_coords
##########################################################################################

def test_uv_from_ra_and_dec_finds_the_body_center(obs: Snapshot) -> None:
    """The direction of Saturn's center maps back to its pixel in the image."""

    backplane = Backplane(obs)
    ra = backplane.center_right_ascension('SATURN')
    dec = backplane.center_declination('SATURN')

    uv = obs.uv_from_ra_and_dec(ra, dec)

    assert not obs.uv_is_outside(uv)
    assert abs(uv.vals[0] - 20.) < 5.
    assert abs(uv.vals[1] - 20.) < 5.


def test_uv_from_ra_and_dec_inverts_the_backplane(obs: Snapshot) -> None:
    """Converting a pixel's direction back to (u,v) returns that pixel."""

    backplane = Backplane(obs)
    ra = backplane.right_ascension()
    dec = backplane.declination()

    # The u-axis of this Snapshot is axis 0 of the array, so the array index
    # (i, j) is the pixel whose (u,v) center is (i + 0.5, j + 0.5)
    uv = obs.uv_from_ra_and_dec(Scalar(ra.vals[10, 30]), Scalar(dec.vals[10, 30]))

    assert uv.vals[0] == pytest.approx(10.5, abs=0.05)
    assert uv.vals[1] == pytest.approx(30.5, abs=0.05)


def test_uv_from_path_finds_the_body_center(obs: Snapshot) -> None:
    """A path is located at the pixel through which it is seen."""

    uv = obs.uv_from_path(Body.lookup('SATURN').path)

    assert not obs.uv_is_outside(uv)


def test_uv_from_path_agrees_with_uv_from_ra_and_dec(obs: Snapshot) -> None:
    """The two routes to the body's center find the same pixel."""

    backplane = Backplane(obs)
    from_radec = obs.uv_from_ra_and_dec(backplane.center_right_ascension('SATURN'),
                                        backplane.center_declination('SATURN'))
    from_path = obs.uv_from_path(Body.lookup('SATURN').path)

    assert from_path.vals == pytest.approx(from_radec.vals, abs=0.1)


def test_uv_from_coords_inverts_the_surface_intercept(obs: Snapshot) -> None:
    """A surface point's coordinates map back to the pixel that sees it."""

    backplane = Backplane(obs)
    surface = Body.lookup('SATURN').surface
    event = backplane.get_surface_event(('SUN<', 'SATURN'))

    unmasked = np.argwhere(event.pos.antimask)
    (i, j) = unmasked[len(unmasked) // 2]
    coords = surface.coords_from_vector3(Vector3(event.pos.vals[i, j]), axes=2)

    uv = obs.uv_from_coords(surface, coords)

    assert uv.vals[0] == pytest.approx(i + 0.5, abs=0.6)
    assert uv.vals[1] == pytest.approx(j + 0.5, abs=0.6)


def test_uv_from_coords_masks_the_far_side(obs: Snapshot) -> None:
    """A point on the underside is masked unless it is asked for."""

    surface = Body.lookup('SATURN').surface
    coords = (Scalar(2.156639432426318), Scalar(-0.5531988322640288))

    assert obs.uv_from_coords(surface, coords).mask
    assert not obs.uv_from_coords(surface, coords, underside=True).mask


##########################################################################################
# inventory
##########################################################################################

def _fresh_obs(solar_system: None) -> Snapshot:
    """A Snapshot identical to the `obs` fixture, but with an empty body cache."""

    return Snapshot(('u', 'v'), TIME, TEXP, FlatFOV((PIXEL, PIXEL), SHAPE),
                    'EARTH', 'TEST_GEOMETRY_CAMERA')


# Of these bodies, only Saturn falls inside the field of view; the others are placed
# outside it by uv_from_path.
INSIDE_BODY = 'SATURN'
OUTSIDE_BODIES = ['MIMAS', 'DIONE', 'TITAN', 'JUPITER']


@pytest.mark.parametrize('name', OUTSIDE_BODIES)
def test_the_other_bodies_really_are_outside_the_field(name: str,
                                                       obs: Snapshot) -> None:
    """Establish the ground truth the inventory tests below are measured against."""

    assert obs.uv_is_outside(obs.uv_from_path(Body.lookup(name).path))


def test_saturn_really_is_inside_the_field(obs: Snapshot) -> None:
    """Establish the ground truth the inventory tests below are measured against."""

    assert not obs.uv_is_outside(obs.uv_from_path(Body.lookup(INSIDE_BODY).path))


def test_inventory_returns_one_flag_per_body(solar_system: None) -> None:
    """"flags" gives one Boolean per body."""

    bodies = [INSIDE_BODY] + OUTSIDE_BODIES
    flags = _fresh_obs(solar_system).inventory(bodies, return_type='flags')

    assert len(flags) == len(bodies)


def test_inventory_returns_a_subset_of_the_bodies(solar_system: None) -> None:
    """The list names bodies that were asked about, and no others."""

    bodies = [INSIDE_BODY] + OUTSIDE_BODIES
    inventory = _fresh_obs(solar_system).inventory(bodies)

    assert set(inventory) <= set(bodies)


def test_inventory_full_describes_every_body(solar_system: None) -> None:
    """"full" gives one entry per body, whether or not it falls inside the FOV."""

    bodies = [INSIDE_BODY] + OUTSIDE_BODIES
    full = _fresh_obs(solar_system).inventory(bodies, return_type='full')

    assert set(full) == set(bodies)


def test_inventory_full_reports_the_body_geometry(solar_system: None) -> None:
    """Each entry carries the body's position, range, and size."""

    entry = _fresh_obs(solar_system).inventory([INSIDE_BODY],
                                               return_type='full')[INSIDE_BODY]

    assert entry['name'] == INSIDE_BODY
    assert entry['range'] == pytest.approx(1.319e9, rel=1.e-3)
    assert entry['outer_radius'] == pytest.approx(60268.)
    assert entry['inner_radius'] == pytest.approx(54364.)
    assert len(entry['center_uv']) == 2


def test_inventory_full_locates_the_body_in_the_field(solar_system: None) -> None:
    """The center of a body inside the FOV matches uv_from_path."""

    observation = _fresh_obs(solar_system)
    entry = observation.inventory([INSIDE_BODY], return_type='full')[INSIDE_BODY]
    expected = observation.uv_from_path(Body.lookup(INSIDE_BODY).path)

    assert entry['center_uv'] == pytest.approx(expected.vals, abs=0.1)


def test_inventory_finds_the_body_that_is_inside(solar_system: None) -> None:
    """Saturn fills much of the field, so it belongs in the inventory."""

    bodies = [INSIDE_BODY] + OUTSIDE_BODIES

    assert INSIDE_BODY in _fresh_obs(solar_system).inventory(bodies)


def test_inventory_rejects_an_unknown_return_type(solar_system: None) -> None:
    """The return type must be "list", "flags", or "full"."""

    with pytest.raises(ValueError, match='invalid return_type'):
        _fresh_obs(solar_system).inventory([INSIDE_BODY], return_type='dict')


@pytest.mark.parametrize('name', OUTSIDE_BODIES)
def test_a_body_outside_the_field_is_not_listed(name: str,
                                                solar_system: None) -> None:
    """Only bodies that fall at least partly inside the FOV are listed."""

    assert _fresh_obs(solar_system).inventory([name]) == []


@pytest.mark.parametrize('bodies', [[INSIDE_BODY, 'MIMAS'],
                                    ['MIMAS', INSIDE_BODY],
                                    ['DIONE', INSIDE_BODY, 'MIMAS'],
                                    [INSIDE_BODY, 'DIONE', 'MIMAS'],
                                    ['MIMAS', 'DIONE', INSIDE_BODY]],
                         ids=lambda names: '+'.join(names))
def test_the_inventory_does_not_depend_on_the_order_of_the_bodies(
        bodies: list, solar_system: None) -> None:
    """The set of bodies inside the FOV is a property of the sky, not of the argument."""

    assert _fresh_obs(solar_system).inventory(bodies) == [INSIDE_BODY]


def test_reordering_the_bodies_on_one_observation_changes_nothing(
        solar_system: None) -> None:
    """One observation gives the same answer however the bodies are ordered, even
    after its body paths have been cached by an earlier call."""

    observation = _fresh_obs(solar_system)

    assert observation.inventory([INSIDE_BODY, 'MIMAS']) == [INSIDE_BODY]
    assert observation.inventory(['MIMAS', INSIDE_BODY]) == [INSIDE_BODY]


def test_the_flags_follow_the_order_of_the_bodies(solar_system: None) -> None:
    """"flags" gives the Booleans in the same order as `bodies`."""

    observation = _fresh_obs(solar_system)
    bodies = ['MIMAS', INSIDE_BODY, 'DIONE']

    assert list(observation.inventory(bodies, return_type='flags')) \
           == [False, True, False]


def test_the_inside_flags_belong_to_their_own_bodies(solar_system: None) -> None:
    """"full" marks each body according to where that body falls."""

    full = _fresh_obs(solar_system).inventory([INSIDE_BODY] + OUTSIDE_BODIES,
                                              return_type='full')

    assert full[INSIDE_BODY]['inside']
    assert not any(full[name]['inside'] for name in OUTSIDE_BODIES)


def test_expanding_the_field_can_admit_a_body(solar_system: None) -> None:
    """A generous expansion pulls more of the sky into the inventory."""

    bodies = [INSIDE_BODY] + OUTSIDE_BODIES
    tight = _fresh_obs(solar_system).inventory(bodies, expand=0.)
    loose = _fresh_obs(solar_system).inventory(bodies, expand=1.e-3)

    assert set(tight) <= set(loose)

##########################################################################################
# uv_from_path on a time-dependent observation
#
# Snapshot overrides uv_from_path, because its pixels all share one time. These exercise
# the iterative solution in the base class, where the pixel found determines the time at
# which the path is evaluated, which in turn determines the pixel.
##########################################################################################

@pytest.fixture(scope='module')
def timed_obs(solar_system: None) -> TimedImage:
    """A TimedImage of Saturn from Earth, its rows swept in time.

    Returns:
        TimedImage: The same camera as `obs`, with the v-axis swept by a Metronome, so
        each row is exposed at a different time.
    """

    rows = Metronome(tstart=TIME, tstride=TEXP/SHAPE[1], texp=TEXP/SHAPE[1],
                     steps=SHAPE[1])

    return TimedImage(('u', 'vt'), rows, FlatFOV((PIXEL, PIXEL), SHAPE), 'EARTH',
                      'TEST_GEOMETRY_CAMERA')


def test_uv_from_path_iterates_to_the_body_center(timed_obs: TimedImage) -> None:
    """The pixel found is inside the field of view, and its own time is consistent."""

    uv = timed_obs.uv_from_path(Body.lookup('SATURN').path)

    assert not timed_obs.uv_is_outside(uv)

    (t0, t1) = timed_obs.time_range_at_uv(uv)
    again = timed_obs.uv_from_path(Body.lookup('SATURN').path, time=0.5 * (t0 + t1))

    assert again.vals == pytest.approx(uv.vals, abs=0.01)


def test_uv_from_path_agrees_with_the_untimed_camera(timed_obs: TimedImage,
                                                     obs: Snapshot) -> None:
    """Sweeping the rows moves the body by well under a pixel over this exposure."""

    swept = timed_obs.uv_from_path(Body.lookup('SATURN').path)
    still = obs.uv_from_path(Body.lookup('SATURN').path)

    assert swept.vals == pytest.approx(still.vals, abs=0.5)


def test_uv_from_path_accepts_overridden_convergence_parameters(
        timed_obs: TimedImage) -> None:
    """Convergence parameters given here override the configured defaults."""

    uv = timed_obs.uv_from_path(Body.lookup('SATURN').path,
                                converge={'max_iterations': 12})

    assert uv == timed_obs.uv_from_path(Body.lookup('SATURN').path)


def test_uv_from_path_reports_a_solution_that_did_not_converge(
        timed_obs: TimedImage, capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind.

    The pixel is still returned, and is still roughly right, which is what the docstring
    promises of a solution that stops early.
    """

    LOGGING.on()
    try:
        uv = timed_obs.uv_from_path(Body.lookup('SATURN').path,
                                    converge={'max_iterations': 1})
    finally:
        LOGGING.off()

    assert 'Observation.uv_from_path did not converge' in capsys.readouterr().out
    assert not timed_obs.uv_is_outside(uv)

def test_uv_from_ra_and_dec_accepts_actual_coordinates(timed_obs: TimedImage) -> None:
    """apparent=False interprets the direction before stellar aberration is applied.

    The aberration of Earth's motion is tens of arcseconds, which is many pixels at this
    scale, so the two interpretations of one direction land in different places.
    """

    event = timed_obs.gridless_event(time=Scalar(timed_obs.midtime))
    (_, arrival) = Body.lookup('SATURN').path.photon_to_event(event)
    (ra, dec) = arrival.ra_and_dec(apparent=False)

    uv = timed_obs.uv_from_ra_and_dec(ra, dec, apparent=False)

    assert not bool(timed_obs.uv_is_outside(uv))
    assert uv != timed_obs.uv_from_ra_and_dec(ra, dec, apparent=True)

##########################################################################################
