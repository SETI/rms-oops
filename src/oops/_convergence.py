##########################################################################################
# oops/_convergence.py: per-ray bookkeeping for iterative solvers
##########################################################################################

import numpy as np

from polymath import Scalar
from oops.config import LOGGING


class RayConvergence:
    """Per-ray convergence bookkeeping for an iteration over an array of rays.

    Each ray is judged on its own. A ray has converged once the size of its step falls
    within its precision goal, and it has failed once its step becomes masked or the
    iterations run out. A ray that has converged or failed takes no further steps, so a
    ray with no solution cannot disturb the solution of any other, and every ray that
    never converges can be masked when the iteration ends.

    Attributes:
        max_change (float): The largest step taken in the latest iteration by any ray that
            was still iterating, or zero if none was; multiply by the km scale to log it.
    """

    def __init__(self, precision, *, mask=None):
        """Constructor for a RayConvergence.

        Parameters:
            precision (ScalarLike): The step size at or below which a ray has converged;
                a single value, or an array broadcastable to the shape of the steps.
            mask (numpy.ndarray | bool, optional): The mask of the rays at the start. A
                ray masked here is never iterated and never counts as a failure. By
                default, the rays masked on the first step are taken to be these; give
                the mask when a ray can become masked on the first step by failing.
        """

        self._precision = Scalar.as_scalar(precision).vals
        self._mask = mask
        self._active = None
        self._converged = None
        self._failed = None
        self.max_change = 0.

    def step(self, dp, precision=None):
        """Record one step and return the step to apply.

        Rays that are still iterating take the full step. A ray whose step is within the
        precision goal is then converged; one whose step is masked has failed.

        Parameters:
            dp (Qube): The full step for every ray, a Scalar or a vector whose length is
                the size of the step; derivatives are preserved.
            precision (ScalarLike, optional): A precision goal for this step alone,
                overriding the one given to the constructor.

        Returns:
            Qube: `dp`, with the step set to zero for rays that were no longer iterating
            when this method was called.
        """

        if precision is None:
            precision = self._precision
        else:
            precision = Scalar.as_scalar(precision).vals

        size = dp.wod.norm() if dp.rank else dp.wod.abs()
        change = np.asarray(size.vals)
        masked = np.broadcast_to(dp.mask, change.shape)

        if self._active is None:
            if self._mask is None:
                self._active = np.logical_not(masked)
            else:
                self._active = np.logical_not(np.broadcast_to(self._mask, change.shape))
            self._converged = np.zeros(change.shape, dtype=np.bool_)
            self._failed = np.zeros(change.shape, dtype=np.bool_)

        active = self._active
        judged = np.logical_and(active, np.logical_not(masked))
        done = np.logical_and(judged, change <= precision)
        lost = np.logical_and(active, masked)

        self._converged = np.logical_or(self._converged, done)
        self._failed = np.logical_or(self._failed, lost)
        self._active = np.logical_and(active, np.logical_not(done | lost))

        self.max_change = float(change[judged].max()) if judged.any() else 0.

        return dp * active.astype(np.float64)

    @property
    def finished(self) -> bool:
        """True if no ray is still iterating.

        Returns:
            bool: True once every ray has converged or failed.
        """

        return self._active is not None and not self._active.any()

    @property
    def failed(self) -> np.ndarray:
        """Mask of the rays that have not converged.

        Returns:
            numpy.ndarray: True for each ray that has failed or is still iterating; the
            rays to mask once the iteration ends.
        """

        if self._active is None:
            return np.zeros((), dtype=np.bool_)

        return np.logical_or(self._failed, self._active)

    @property
    def failures(self) -> int:
        """The number of rays that have not converged.

        Returns:
            int: The count of True values in `failed`.
        """

        return int(np.count_nonzero(self.failed))


def limited_step(dp, p, lower):
    """A Newton step limited to half the distance from the current value to a bound.

    Parameters:
        dp (Scalar): The full Newton step for every ray, to be subtracted from `p`.
        p (Scalar): The current value for every ray.
        lower (float): The bound below which the values must not go.

    Returns:
        Scalar: `dp`, reduced wherever the full step would carry `p` more than halfway
        to `lower`.
    """

    limit = 0.5 * (p - lower)
    over = np.asarray(dp.wod.vals > limit.wod.vals)
    if not over.any():
        return dp

    over = over.astype(np.float64)
    return dp * (1. - over) + limit * over


def iterate(newton_step, start, precision, max_iterations, *, name, km_scale=1.,
            debug=False):
    """Newton's method from the given starting values, judging each ray on its own.

    Parameters:
        newton_step (Callable[[Scalar], Scalar]): Function returning the Newton step
            `f / df_dp` for every ray from the current values.
        start (Scalar): The starting value for every ray; masked rays are not iterated.
        precision (ScalarLike): The step size at or below which a ray has converged.
        max_iterations (int): The maximum number of steps to take.
        name (str): The name of the calling method, for the convergence log.
        km_scale (float, optional): Factor converting a step to km, for the log.
        debug (bool, optional): True to log every iteration even if the convergence
            category of :class:`~oops.config.LOGGING` is off.

    Returns:
        tuple[Scalar, numpy.ndarray]: The converged values, unmasked and unchanged for
        the rays that did not converge, and the mask of those rays.
    """

    convergence = RayConvergence(precision)
    p = start
    for count in range(max_iterations):
        p = p - convergence.step(newton_step(p))

        if LOGGING.surface_iterations or debug:
            LOGGING.convergence(f'{name}(): iter={count+1}; '
                                f'change[km]={convergence.max_change * km_scale:.6g}')

        if convergence.finished:
            break

    return (p, convergence.failed)


def solve(newton_step, estimate, guess, precision, max_iterations, *, name, km_scale=1.,
          debug=False):
    """Newton's method from a supplied guess, falling back on a default estimate.

    The iteration starts from `guess` wherever one is supplied and unmasked, and from
    `estimate` elsewhere. A guess carried over from a different geometry can start the
    iteration far from any solution, so wherever the iteration from the guess fails, it is
    repeated from the estimate. Rays that still fail are masked and a warning is logged.

    Parameters:
        newton_step (Callable[[Scalar], Scalar]): Function returning the Newton step
            `f / df_dp` for every ray from the current values.
        estimate (Scalar): The default starting value for every ray.
        guess (Scalar | None): A supplied starting value, broadcastable to `estimate`, or
            None.
        precision (ScalarLike): The step size at or below which a ray has converged.
        max_iterations (int): The maximum number of steps to take from each start.
        name (str): The name of the calling method, for the convergence log and the
            warning.
        km_scale (float, optional): Factor converting a step to km, for the log.
        debug (bool, optional): True to log every iteration even if the convergence
            category of :class:`~oops.config.LOGGING` is off.

    Returns:
        Scalar: The converged values, masked where the iteration did not converge.
    """

    if guess is None:
        start = estimate
    else:
        (estimate_vals, guess_vals) = np.broadcast_arrays(estimate.vals, guess.vals)
        use_guess = np.logical_not(np.broadcast_to(guess.mask, guess_vals.shape))
        start = Scalar(np.where(use_guess, guess_vals, estimate_vals), mask=estimate.mask)

    (p, failed) = iterate(newton_step, start, precision, max_iterations, name=name,
                          km_scale=km_scale, debug=debug)

    if guess is not None and failed.any():
        retry = Scalar(np.broadcast_to(estimate.vals, p.shape),
                       mask=np.logical_or(np.broadcast_to(estimate.mask, p.shape),
                                          np.logical_not(failed)))
        (p_retry, failed_retry) = iterate(newton_step, retry, precision, max_iterations,
                                          name=name, km_scale=km_scale, debug=debug)
        p = Scalar(np.where(failed, p_retry.vals, p.vals), mask=p.mask)
        failed = np.logical_and(failed, failed_retry)

    if failed.any():
        LOGGING.warn(f'{name}() did not converge: '
                     f'rays={np.count_nonzero(failed)}/{failed.size}')
        p = p.remask_or(failed)

    return p

##########################################################################################
