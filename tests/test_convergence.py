##########################################################################################
# tests/test_convergence.py
##########################################################################################

import numpy as np
import pytest

from polymath          import Scalar
from oops.config       import LOGGING
from oops._convergence import RayConvergence, iterate, limited_step, solve


def test_a_ray_converges_when_its_step_is_within_precision() -> None:
    """A step at or below the precision marks the ray converged and stops its iteration."""

    convergence = RayConvergence(1.e-3)

    convergence.step(Scalar([1., 1.]))
    step = convergence.step(Scalar([1.e-3, 0.5]))

    assert step.vals.tolist() == [1.e-3, 0.5]
    assert convergence.failed.tolist() == [False, True]
    assert not convergence.finished


def test_a_converged_ray_takes_no_further_steps() -> None:
    """Once converged, a ray's later steps are zeroed while the others continue."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1.e-4, 1.]))

    step = convergence.step(Scalar([5., 0.1]))

    assert step.vals.tolist() == [0., 0.1]


def test_a_zeroed_step_keeps_its_derivatives_zeroed() -> None:
    """The derivatives of a zeroed step are zeroed too."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1.e-4, 1.]))
    dp = Scalar([5., 0.1])
    dp.insert_deriv('x', Scalar([7., 7.]))

    step = convergence.step(dp)

    assert step.d_dx.vals.tolist() == [0., 7.]


def test_a_growing_step_keeps_a_ray_iterating() -> None:
    """A step larger than the previous one does not end a ray's iteration."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1., 1.]))
    step = convergence.step(Scalar([2., 0.5]))

    assert step.vals.tolist() == [2., 0.5]
    assert convergence.failed.tolist() == [True, True]
    assert not convergence.finished


def test_a_ray_that_fails_does_not_stop_the_others() -> None:
    """The other rays keep iterating after one has failed."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1., 1.]))
    convergence.step(Scalar([2., 0.5], mask=[True, False]))
    convergence.step(Scalar([2., 1.e-4]))

    assert convergence.failed.tolist() == [True, False]
    assert convergence.finished


def test_a_ray_that_becomes_masked_has_failed() -> None:
    """A masked step marks a ray that was iterating as failed."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1., 1.]))
    convergence.step(Scalar([0.5, 0.5], mask=[True, False]))
    convergence.step(Scalar([0.1, 1.e-4]))

    assert convergence.failed.tolist() == [True, False]


def test_a_ray_masked_from_the_start_is_never_iterated() -> None:
    """A ray masked on the first step neither iterates nor counts as a failure."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1., 1.e-4], mask=[True, False]))

    assert convergence.failed.tolist() == [False, False]
    assert convergence.failures == 0
    assert convergence.finished


def test_a_ray_masked_on_the_first_step_has_failed_if_it_started_unmasked() -> None:
    """With the starting mask given, a first step that comes back masked is a failure."""

    convergence = RayConvergence(1.e-3, mask=np.array([True, False]))
    convergence.step(Scalar([1., 1.], mask=[True, True]))

    assert convergence.failed.tolist() == [False, True]
    assert convergence.finished


def test_the_largest_step_is_reported_over_the_iterating_rays() -> None:
    """max_change covers the rays that were iterating, not the ones already converged."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar([1.e-4, 1.]))
    convergence.step(Scalar([9., 0.5]))

    assert convergence.max_change == 0.5


def test_the_precision_may_vary_by_ray() -> None:
    """Each ray is judged against its own precision."""

    convergence = RayConvergence(Scalar([1.e-2, 1.e-4]))
    convergence.step(Scalar([1.e-3, 1.e-3]))

    assert convergence.failed.tolist() == [False, True]


def test_a_shapeless_step_is_handled() -> None:
    """A single ray works the same as an array of them."""

    convergence = RayConvergence(1.e-3)
    convergence.step(Scalar(1.))
    step = convergence.step(Scalar(1.e-4))

    assert step.vals == 1.e-4
    assert convergence.finished
    assert convergence.failures == 0


def _square_root_step(p: Scalar) -> Scalar:
    """The Newton step for f(p) = p**2 - 4, whose roots are +2 and -2."""

    return Scalar.as_scalar((p * p - 4.) / (2. * p))


def test_iterate_converges_each_ray_from_its_start() -> None:
    """iterate() reaches the root nearest each starting value."""

    (p, failed) = iterate(_square_root_step, Scalar([3., -3.]), 1.e-9, 20, name='test')

    assert p.vals == pytest.approx([2., -2.])
    assert failed.tolist() == [False, False]


def test_iterate_reports_the_rays_that_run_out_of_iterations() -> None:
    """A ray that is still iterating at the limit is reported as failed."""

    (p, failed) = iterate(_square_root_step, Scalar([3., 1.e6]), 1.e-9, 8, name='test')

    assert failed.tolist() == [False, True]
    assert not p.mask


def test_a_limited_step_stops_halfway_to_the_bound() -> None:
    """A step that would cross the bound is cut to half the distance to it."""

    p = Scalar([1., 1.])
    step = limited_step(Scalar([5., 0.5]), p, -1.)

    assert step.vals.tolist() == [1., 0.5]


def test_a_limited_step_keeps_its_derivatives() -> None:
    """The derivatives of the limited step follow those of the current value."""

    p = Scalar([1., 1.])
    p.insert_deriv('x', Scalar([2., 2.]))
    dp = Scalar([5., 0.5])
    dp.insert_deriv('x', Scalar([3., 3.]))

    step = limited_step(dp, p, -1.)

    assert step.d_dx.vals.tolist() == [1., 3.]


def test_solve_starts_from_the_guess_where_there_is_one() -> None:
    """The guess selects the root, and the estimate is used where the guess is masked."""

    guess = Scalar([-3., 3.], mask=[False, True])

    p = solve(_square_root_step, Scalar([3., -3.]), guess, 1.e-9, 20, name='test')

    assert p.vals == pytest.approx([-2., -2.])


def test_solve_retries_from_the_estimate_where_the_guess_fails() -> None:
    """A guess too far from a root to converge in time is replaced by the estimate."""

    LOGGING.reset()
    p = solve(_square_root_step, Scalar([3., 3.]), Scalar([3., 1.e6]), 1.e-9, 6,
              name='test')

    assert LOGGING.warnings == 0
    assert p.vals == pytest.approx([2., 2.])
    assert not np.any(p.mask)


def test_solve_masks_the_rays_that_never_converge_and_warns() -> None:
    """A ray that fails from both starts is masked, and one warning is logged."""

    LOGGING.reset()
    p = solve(_square_root_step, Scalar([3., 1.e6]), Scalar([3., 1.e6]), 1.e-9, 8,
              name='test')

    assert LOGGING.warnings == 1
    assert p.mask.tolist() == [False, True]
    assert p.vals[0] == pytest.approx(2.)

##########################################################################################
