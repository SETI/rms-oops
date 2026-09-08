##########################################################################################
# tests/surface/test_photon_solver_options.py: the options shared by the photon solvers
#
# Every solver in oops/surface/_photon_solver.py takes the same set of options: an
# antimask, an initial guess, convergence overrides, and derivatives. These tests
# exercise those options across all four solvers.
##########################################################################################

from collections.abc import Iterator

import numpy as np
import pytest

from polymath               import Scalar, Vector3
from oops.config            import LOGGING
from oops.event             import Event
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.path.spicepath    import SpicePath
from oops.surface.ansa      import Ansa
from oops.surface.ellipsoid import Ellipsoid

REQ = 6378.
RPOL = 6357.

# Two observers well outside the planet, so a solution exists for each
OBSERVER = Vector3([(1.e6, 0., 1.e5), (0., 1.e6, -1.e5)])


@pytest.fixture
def planet() -> Ellipsoid:
    """An oblate Ellipsoid centered on the SSB, so its origin does not move.

    Returns:
        Ellipsoid: The surface every solver here is applied to.
    """

    return Ellipsoid(Path.SSB, Frame.J2000, (REQ, REQ, RPOL))


def _arrival() -> Event:
    """An arrival event at the observers, receiving photons from the planet.

    Returns:
        Event: The event, with its arrival directions filled in.
    """

    event = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)
    event.arr = OBSERVER                # the photon travels outward, toward the observer

    return event


def _coords(planet: Ellipsoid) -> tuple:
    """The surface coordinates of the points the line-of-sight solver finds.

    Parameters:
        planet: The surface.

    Returns:
        tuple: The two surface coordinates of each intercept.
    """

    (surface_event, _) = planet.photon_to_event(_arrival())

    return planet.coords_from_vector3(surface_event.pos, axes=2)


##########################################################################################
# The line-of-sight solver
##########################################################################################

def test_convergence_parameters_can_be_overridden(planet: Ellipsoid) -> None:
    """Parameters given here replace the configured defaults, one key at a time."""

    (few, _) = planet.photon_to_event(_arrival(), converge={'max_iterations': 2})
    (many, _) = planet.photon_to_event(_arrival())

    assert few.pos.vals == pytest.approx(many.pos.vals, abs=1.e-6)


def test_a_fully_masked_guess_is_discarded(planet: Ellipsoid) -> None:
    """A guess with nothing usable in it is dropped in favor of the default."""

    guess = Scalar([0., 0.], True)

    (with_guess, _) = planet.photon_to_event(_arrival(), guess=guess)
    (without, _) = planet.photon_to_event(_arrival())

    assert with_guess.pos == without.pos


def test_a_partly_masked_guess_is_filled_in(planet: Ellipsoid) -> None:
    """A masked element of the guess is replaced by the mean of the rest."""

    (reference, _) = planet.photon_to_event(_arrival())
    guess = Scalar(reference.time.vals, [False, True])

    (with_guess, _) = planet.photon_to_event(_arrival(), guess=guess)

    assert with_guess.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-6)


def test_an_antimask_is_combined_with_the_mask_of_the_link(planet: Ellipsoid) -> None:
    """An element masked in either the antimask or the link drops out of the solution."""

    arrival = _arrival()
    arrival = arrival.mask_where(np.array([False, True]))

    (surface_event, _) = planet.photon_to_event(arrival, antimask=np.array([True, True]))

    assert list(surface_event.pos.mask) == [False, True]


def test_derivatives_of_the_link_are_restored_to_the_solution(planet: Ellipsoid) -> None:
    """With derivs, the derivatives of the arrival event survive into the result."""

    arrival = _arrival()
    arrival.arr.insert_deriv('los', Vector3.IDENTITY)

    (surface_event, _) = planet.photon_to_event(arrival, derivs=True)

    assert 'los' in surface_event.pos.derivs


def test_the_iterations_of_the_line_of_sight_solver_can_be_logged(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    LOGGING.on()
    try:
        planet.photon_to_event(_arrival())
    finally:
        LOGGING.off()

    assert 'Ellipsoid._solve_photon_by_los' in capsys.readouterr().out


def test_a_line_of_sight_solution_that_runs_out_of_iterations_is_reported(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind."""

    LOGGING.on()
    try:
        (surface_event, _) = planet.photon_to_event(_arrival(),
                                                    converge={'max_iterations': 1})
    finally:
        LOGGING.off()

    assert 'did not converge' in capsys.readouterr().out
    assert not np.any(surface_event.pos.mask)


##########################################################################################
# The coordinate solver
##########################################################################################

def test_the_coordinate_solver_accepts_overridden_convergence(planet: Ellipsoid) -> None:
    """Parameters given here replace the configured defaults."""

    coords = _coords(planet)

    (few, _) = planet.photon_to_coords(_arrival(), coords,
                                       converge={'max_iterations': 2})
    (many, _) = planet.photon_to_coords(_arrival(), coords)

    assert few.pos.vals == pytest.approx(many.pos.vals, abs=1.e-3)


def test_the_coordinate_solver_discards_a_fully_masked_guess(planet: Ellipsoid) -> None:
    """A guess with nothing usable in it is dropped in favor of the default."""

    coords = _coords(planet)
    guess = Scalar([0., 0.], True)

    (with_guess, _) = planet.photon_to_coords(_arrival(), coords, guess=guess)
    (without, _) = planet.photon_to_coords(_arrival(), coords)

    assert with_guess.pos == without.pos


def test_the_coordinate_solver_fills_in_a_partly_masked_guess(planet: Ellipsoid) -> None:
    """A masked element of the guess is replaced by the mean of the rest."""

    coords = _coords(planet)
    (reference, _) = planet.photon_to_coords(_arrival(), coords)
    guess = Scalar(reference.time.vals, [False, True])

    (with_guess, _) = planet.photon_to_coords(_arrival(), coords, guess=guess)

    assert with_guess.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-3)


def test_the_coordinate_solver_applies_the_antimask(planet: Ellipsoid) -> None:
    """An element the antimask excludes drops out of the solution."""

    coords = _coords(planet)

    (surface_event, _) = planet.photon_to_coords(_arrival(), coords,
                                                 antimask=np.array([True, False]))

    assert list(surface_event.pos.mask) == [False, True]


def test_the_coordinate_solver_masks_an_entirely_masked_link(planet: Ellipsoid) -> None:
    """With nothing left to solve for, the result is entirely masked."""

    coords = _coords(planet)

    (surface_event, _) = planet.photon_to_coords(_arrival(), coords,
                                                 antimask=np.array([False, False]))

    assert np.all(surface_event.pos.mask)


def test_the_iterations_of_the_coordinate_solver_can_be_logged(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    coords = _coords(planet)

    LOGGING.on()
    try:
        planet.photon_to_coords(_arrival(), coords)
    finally:
        LOGGING.off()

    assert 'Surface._solve_photon_by_coords' in capsys.readouterr().out


def test_a_coordinate_solution_that_runs_out_of_iterations_is_reported(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind."""

    coords = _coords(planet)

    LOGGING.on()
    try:
        planet.photon_to_coords(_arrival(), coords, converge={'max_iterations': 0})
    finally:
        LOGGING.off()

    assert 'did not converge' in capsys.readouterr().out


def test_the_coordinate_solver_handles_a_virtual_surface() -> None:
    """An ansa surface is virtual, so its position depends on the observer."""

    ansa = Ansa(Path.SSB, Frame.J2000)
    obs = Vector3([(1.e6, 0., 0.), (0., 1.e6, 0.)])
    arrival = Event(Scalar(0.), obs, Path.SSB, Frame.J2000)
    arrival.arr = obs
    coords = (Scalar([1.e5, 1.e5]), Scalar([0., 0.]))

    (surface_event, _) = ansa.photon_to_coords(arrival, coords)

    assert surface_event.pos.norm().vals == pytest.approx([1.e5, 1.e5], rel=1.e-6)


##########################################################################################
# The solver based on the surface normal and a remote event
##########################################################################################

def test_a_virtual_surface_has_no_normal_solution() -> None:
    """The normal solvers need a real surface, not a virtual one."""

    ansa = Ansa(Path.SSB, Frame.J2000)
    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    with pytest.raises(ValueError, match='does not support virtual surface class Ansa'):
        ansa.photon_normal_to_event(arrival)


def test_the_normal_solver_accepts_overridden_convergence(planet: Ellipsoid) -> None:
    """Parameters given here replace the configured defaults."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    (few, _) = planet.photon_normal_to_event(arrival, converge={'max_iterations': 3})
    (many, _) = planet.photon_normal_to_event(arrival)

    assert few.pos.vals == pytest.approx(many.pos.vals, abs=1.e-6)


def test_the_normal_solver_discards_a_fully_masked_guess(planet: Ellipsoid) -> None:
    """A guess with nothing usable in it is dropped in favor of the default."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)
    guess = Scalar([0., 0.], True)

    (with_guess, _) = planet.photon_normal_to_event(arrival, guess=guess)
    (without, _) = planet.photon_normal_to_event(arrival)

    assert with_guess.pos == without.pos


def test_the_normal_solver_fills_in_a_partly_masked_guess(planet: Ellipsoid) -> None:
    """A masked element of the guess is replaced by the mean of the rest."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)
    (reference, _) = planet.photon_normal_to_event(arrival)
    guess = Scalar(reference.time.vals, [False, True])

    (with_guess, _) = planet.photon_normal_to_event(arrival, guess=guess)

    assert with_guess.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-6)


def test_the_normal_solver_applies_the_antimask(planet: Ellipsoid) -> None:
    """An element the antimask excludes drops out of the solution."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    (surface_event, _) = planet.photon_normal_to_event(arrival,
                                                       antimask=np.array([True, False]))

    assert list(surface_event.pos.mask) == [False, True]


def test_the_normal_solver_masks_an_entirely_masked_link(planet: Ellipsoid) -> None:
    """With nothing left to solve for, the result is entirely masked."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    (surface_event, _) = planet.photon_normal_to_event(arrival,
                                                       antimask=np.array([False, False]))

    assert np.all(surface_event.pos.mask)


def test_the_iterations_of_the_normal_solver_can_be_logged(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    LOGGING.on()
    try:
        planet.photon_normal_to_event(arrival)
    finally:
        LOGGING.off()

    assert 'Surface._solve_photon_event_normal' in capsys.readouterr().out


def test_a_normal_solution_that_runs_out_of_iterations_is_reported(
        planet: Ellipsoid, capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind."""

    arrival = Event(Scalar(0.), OBSERVER, Path.SSB, Frame.J2000)

    LOGGING.on()
    try:
        planet.photon_normal_to_event(arrival, converge={'max_iterations': 1})
    finally:
        LOGGING.off()

    assert 'did not converge' in capsys.readouterr().out


##########################################################################################
# The solver based on the surface normal and a remote path
##########################################################################################

@pytest.fixture
def sun_and_planet(core_kernels) -> Iterator[tuple]:
    """The Sun as a SpicePath, with an Earth-sized Ellipsoid centered on the SSB.

    Yields:
        tuple: The Sun's path and the planet's surface.
    """

    yield (SpicePath('SUN', 'SSB'), Ellipsoid(Path.SSB, Frame.J2000, (REQ, REQ, RPOL)))


# Two times, so an antimask can drop one of them
TIMES = Scalar([0., 100.])


def test_a_virtual_surface_has_no_path_normal_solution() -> None:
    """The normal solvers need a real surface, not a virtual one."""

    ansa = Ansa(Path.SSB, Frame.J2000)

    with pytest.raises(ValueError, match='does not support virtual surface class Ansa'):
        ansa.photon_path_to_normal(TIMES, Path.SSB)


def test_the_path_normal_solver_accepts_overridden_convergence(sun_and_planet) -> None:
    """Parameters given here replace the configured defaults."""

    (sun, planet) = sun_and_planet

    (few, _) = planet.photon_path_to_normal(TIMES, sun, converge={'max_iterations': 3})
    (many, _) = planet.photon_path_to_normal(TIMES, sun)

    assert few.pos.vals == pytest.approx(many.pos.vals, abs=1.e-6)


def test_the_path_normal_solver_discards_a_fully_masked_guess(sun_and_planet) -> None:
    """A guess with nothing usable in it is dropped in favor of the default."""

    (sun, planet) = sun_and_planet
    guess = Scalar([0., 0.], True)

    (with_guess, _) = planet.photon_path_to_normal(TIMES, sun, guess=guess)
    (without, _) = planet.photon_path_to_normal(TIMES, sun)

    assert with_guess.pos == without.pos


def test_the_path_normal_solver_fills_in_a_partly_masked_guess(sun_and_planet) -> None:
    """A masked element of the guess is replaced by the mean of the rest."""

    (sun, planet) = sun_and_planet
    (_, reference_path) = planet.photon_path_to_normal(TIMES, sun)
    guess = Scalar(reference_path.time.vals, [False, True])

    (surface_event, _) = planet.photon_path_to_normal(TIMES, sun, guess=guess)
    (reference, _) = planet.photon_path_to_normal(TIMES, sun)

    assert surface_event.pos.vals == pytest.approx(reference.pos.vals, abs=1.e-2)


def test_the_path_normal_solver_combines_the_antimask_with_the_time(
        sun_and_planet) -> None:
    """An antimask that keeps everything leaves the solution unchanged."""

    (sun, planet) = sun_and_planet

    (with_all, _) = planet.photon_path_to_normal(TIMES, sun,
                                                 antimask=np.array([True, True]))
    (plain, _) = planet.photon_path_to_normal(TIMES, sun)

    assert with_all.pos == plain.pos


def test_the_path_normal_solver_masks_an_entirely_masked_time(sun_and_planet) -> None:
    """With nothing left to solve for, the result is entirely masked."""

    (sun, planet) = sun_and_planet

    (surface_event, _) = planet.photon_path_to_normal(TIMES, sun,
                                                      antimask=np.array([False, False]))

    assert np.all(surface_event.pos.mask)


def test_the_iterations_of_the_path_normal_solver_can_be_logged(
        sun_and_planet, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    (sun, planet) = sun_and_planet

    LOGGING.on()
    try:
        planet.photon_path_to_normal(TIMES, sun)
    finally:
        LOGGING.off()

    assert 'Surface._solve_photon_path_normal' in capsys.readouterr().out


def test_a_path_normal_solution_that_runs_out_of_iterations_is_reported(
        sun_and_planet, capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind."""

    (sun, planet) = sun_and_planet

    LOGGING.on()
    try:
        planet.photon_path_to_normal(TIMES, sun, converge={'max_iterations': 1})
    finally:
        LOGGING.off()

    assert 'did not converge' in capsys.readouterr().out


##########################################################################################
