##########################################################################################
# tests/path/test_photon_solver.py: the light-time solver of the Path class
##########################################################################################

import numpy as np
import pytest

from polymath          import Scalar, Vector3
from oops.config       import LOGGING
from oops.constants    import C
from oops.event        import Event
from oops.frame.frame_ import Frame
from oops.path.path_   import Path
from oops.path.fixedpath  import FixedPath
from oops.path.linearpath import LinearPath

# A target one light-second away along the x-axis, so the light travel time is exactly
# one second and nothing moves during it
RANGE = C
TIMES = Scalar([0., 100.])


@pytest.fixture
def target() -> FixedPath:
    """A point at rest one light-second from the barycenter.

    Returns:
        FixedPath: The target path.
    """

    return FixedPath(Vector3((RANGE, 0., 0.)), Path.SSB, Frame.J2000,
                     path_id='TEST_PHOTON_TARGET')


def _arrival(times=TIMES) -> Event:
    """An arrival event at the barycenter.

    Parameters:
        times: The times of the arrival, in seconds TDB.

    Returns:
        Event: The event, at rest at the origin.
    """

    return Event(times, Vector3.ZERO, Path.SSB, Frame.J2000)


def test_the_light_travel_time_is_the_range_divided_by_the_speed_of_light(
        target: FixedPath) -> None:
    """A target one light-second away is seen one second in the past."""

    (path_event, arrival) = target.photon_to_event(_arrival())

    assert path_event.dep_lt.vals == pytest.approx([1., 1.], abs=1.e-9)
    assert arrival.arr_lt.vals == pytest.approx([-1., -1.], abs=1.e-9)
    assert path_event.time.vals == pytest.approx([-1., 99.], abs=1.e-9)


def test_a_departing_photon_reaches_the_target_one_light_time_later(
        target: FixedPath) -> None:
    """The same range gives the same travel time in the other direction."""

    departure = _arrival()

    (path_event, departure_event) = target.photon_from_event(departure)

    assert path_event.arr_lt.vals == pytest.approx([-1., -1.], abs=1.e-9)
    assert departure_event.dep_lt.vals == pytest.approx([1., 1.], abs=1.e-9)
    assert path_event.time.vals == pytest.approx([1., 101.], abs=1.e-9)


def test_an_initial_guess_reaches_the_same_solution(target: FixedPath) -> None:
    """A guess at the path event time only starts the iteration off."""

    (reference, _) = target.photon_to_event(_arrival())
    guess = Scalar(reference.time.vals)

    (path_event, _) = target.photon_to_event(_arrival(), guess=guess)

    assert path_event.time.vals == pytest.approx(reference.time.vals, abs=1.e-9)


def test_a_shapeless_guess_is_broadcast_to_the_shape_of_the_link(
        target: FixedPath) -> None:
    """One guess serves an array of link times."""

    (reference, _) = target.photon_to_event(_arrival())

    (path_event, _) = target.photon_to_event(_arrival(), guess=Scalar(0.))

    assert path_event.shape == (2,)
    assert path_event.time.vals == pytest.approx(reference.time.vals, abs=1.e-9)


def test_an_entirely_masked_link_gives_an_entirely_masked_result(
        target: FixedPath) -> None:
    """With nothing to solve for, both returned events are masked."""

    (path_event, arrival) = target.photon_to_event(_arrival(),
                                                   antimask=np.array([False, False]))

    assert np.all(path_event.mask)
    assert np.all(arrival.mask)


def test_an_entirely_masked_link_keeps_its_derivatives(target: FixedPath) -> None:
    """With derivs, the masked result still carries the derivatives its caller expects."""

    (path_event, arrival) = target.photon_to_event(_arrival(), derivs=True,
                                                   antimask=np.array([False, False]))

    assert 't' in arrival.arr_lt.derivs
    assert 'los' in arrival.arr_lt.derivs
    assert np.all(path_event.mask)


def test_the_iterations_of_the_solver_can_be_logged(
        target: FixedPath, capsys: pytest.CaptureFixture[str]) -> None:
    """Each pass of the solver reports the change it made."""

    LOGGING.on()
    try:
        target.photon_to_event(_arrival())
    finally:
        LOGGING.off()

    assert 'Path._solve_photon' in capsys.readouterr().out


def test_a_solution_that_runs_out_of_iterations_is_reported(
        capsys: pytest.CaptureFixture[str]) -> None:
    """Capping the iterations below what is needed leaves a warning behind.

    A moving target needs more than one pass, because the light time depends on where
    the target was when the photon left it.
    """

    moving = LinearPath((Vector3((RANGE, 0., 0.)), Vector3((1000., 0., 0.))), 0.,
                        Path.SSB, frame=Frame.J2000, path_id='TEST_PHOTON_MOVING')

    LOGGING.on()
    try:
        (path_event, _) = moving.photon_to_event(_arrival(),
                                                 converge={'max_iterations': 1})
    finally:
        LOGGING.off()

    assert 'did not converge' in capsys.readouterr().out

    # The solution it stops at is still within a few microseconds of the converged one
    (converged, _) = moving.photon_to_event(_arrival())
    assert path_event.dep_lt.vals == pytest.approx(converged.dep_lt.vals, abs=1.e-4)

##########################################################################################
