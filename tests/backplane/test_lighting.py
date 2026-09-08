##########################################################################################
# tests/backplane/test_lighting.py
##########################################################################################

import numpy as np
import pytest

from oops.backplane import Backplane

PLANET = 'SATURN'
RING = 'SATURN:RING'
PI = np.pi


def _unmasked(array) -> np.ndarray:
    """The values of a backplane where it is not masked."""

    return array.vals[array.antimask]


def test_incidence_angle_runs_from_zero_to_pi(bp: Backplane) -> None:
    """The incidence angle is measured from the surface normal."""

    values = _unmasked(bp.incidence_angle(PLANET))

    assert np.all(values >= 0.)
    assert np.all(values <= PI)


def test_emission_angle_runs_from_zero_to_pi(bp: Backplane) -> None:
    """The emission angle is measured from the surface normal."""

    values = _unmasked(bp.emission_angle(PLANET))

    assert np.all(values >= 0.)
    assert np.all(values <= PI)


def test_phase_angle_runs_from_zero_to_pi(bp: Backplane) -> None:
    """The phase angle separates the arriving and departing photons."""

    values = _unmasked(bp.phase_angle(PLANET))

    assert np.all(values >= 0.)
    assert np.all(values <= PI)


def test_phase_angle_of_saturn_from_earth_is_small(bp: Backplane) -> None:
    """Seen from Earth, an outer planet is always close to full phase."""

    assert np.all(_unmasked(bp.phase_angle(PLANET)) < 0.2)


def test_scattering_angle_is_the_supplement_of_the_phase_angle(bp: Backplane) -> None:
    """The two angles are measured from opposite ends of the same line."""

    phase = bp.phase_angle(PLANET)
    scattering = bp.scattering_angle(PLANET)

    assert np.allclose(_unmasked(phase) + _unmasked(scattering), PI)


def test_mu0_is_the_cosine_of_the_incidence_angle(bp: Backplane) -> None:
    """mu0 = cos(incidence_angle)."""

    assert np.allclose(_unmasked(bp.mu0(PLANET)),
                       np.cos(_unmasked(bp.incidence_angle(PLANET))))


def test_mu_is_the_cosine_of_the_emission_angle(bp: Backplane) -> None:
    """mu = cos(emission_angle)."""

    assert np.allclose(_unmasked(bp.mu(PLANET)),
                       np.cos(_unmasked(bp.emission_angle(PLANET))))


def test_lambert_law_is_the_cosine_of_the_incidence_angle(bp: Backplane) -> None:
    """The Lambert law model is cos(incidence_angle), clipped at zero."""

    lambert = _unmasked(bp.lambert_law(PLANET))
    mu0 = _unmasked(bp.mu0(PLANET))

    assert np.allclose(lambert, np.clip(mu0, 0., None))


def test_lambert_law_is_never_negative(bp: Backplane) -> None:
    """An unlit surface reflects nothing."""

    assert np.all(_unmasked(bp.lambert_law(PLANET)) >= 0.)


def test_lommel_seeliger_law_is_mu0_over_mu_plus_mu0(bp: Backplane) -> None:
    """The Lommel-Seeliger model is mu0 / (mu + mu0)."""

    mu = _unmasked(bp.mu(PLANET))
    mu0 = _unmasked(bp.mu0(PLANET))
    expected = np.clip(mu0, 0., None) / (np.clip(mu, 0., None)
                                         + np.clip(mu0, 0., None))

    assert np.allclose(_unmasked(bp.lommel_seeliger_law(PLANET)), expected,
                       equal_nan=True)


def test_minnaert_law_with_exponent_one_is_lambert_over_nothing(bp: Backplane) -> None:
    """With k=1 and k2=0, the Minnaert model reduces to mu0."""

    minnaert = bp.minnaert_law(PLANET, 1., k2=0.)
    mu0 = bp.mu0(PLANET)
    antimask = minnaert.antimask

    assert np.allclose(minnaert.vals[antimask], np.clip(mu0.vals[antimask], 0., None))


def test_minnaert_law_clips_the_emission_cosine(bp: Backplane) -> None:
    """A higher clip limit changes the model only near the limb, where mu is small."""

    loose = bp.minnaert_law(PLANET, 0.7, clip=0.05)
    tight = bp.minnaert_law(PLANET, 0.7, clip=0.5)

    assert np.any(loose.vals != tight.vals)


def test_incidence_angle_is_apparent_by_default(bp: Backplane) -> None:
    """The default accounts for aberration in the surface frame."""

    apparent = bp.incidence_angle(PLANET, apparent=True)
    actual = bp.incidence_angle(PLANET, apparent=False)

    assert np.any(_unmasked(apparent) != _unmasked(actual))
    assert bp.incidence_angle(PLANET) is apparent


def test_aberration_shifts_the_angles_only_slightly(bp: Backplane) -> None:
    """The correction is small compared with the angles themselves."""

    apparent = _unmasked(bp.incidence_angle(PLANET, apparent=True))
    actual = _unmasked(bp.incidence_angle(PLANET, apparent=False))

    assert np.max(np.abs(apparent - actual)) < 1.e-3


@pytest.mark.parametrize('method', ['center_incidence_angle', 'center_emission_angle',
                                    'center_phase_angle', 'center_scattering_angle'])
def test_center_angles_are_gridless(method: str, bp: Backplane) -> None:
    """A center backplane refers to the body's path, so it has no spatial extent."""

    assert getattr(bp, method)(PLANET).shape == ()


@pytest.mark.parametrize('body', [PLANET, RING])
def test_center_emission_angle_does_not_depend_on_evaluation_order(
        body: str, fresh_bp: Backplane, bp: Backplane) -> None:
    """The center emission angle is the same from an empty cache as from a warm one.

    A gridless event is filed on the body's path rather than on its surface, without the
    surface subfield that the ring branch of the emission angle consults. The surface has
    to be looked up by name; reading it off the event succeeds only once some other
    backplane has filled that subfield in, which is what makes the order matter.
    """

    assert fresh_bp.center_emission_angle(body).vals == pytest.approx(
                                                    bp.center_emission_angle(body).vals)


def test_center_phase_angle_matches_the_disk(bp: Backplane) -> None:
    """The phase angle at the center falls within the range across the disk."""

    values = _unmasked(bp.phase_angle(PLANET))
    center = bp.center_phase_angle(PLANET)

    assert values.min() <= center.vals <= values.max()


def test_center_scattering_is_the_supplement_of_center_phase(bp: Backplane) -> None:
    """The relation holds at the body's center as it does across the disk."""

    total = bp.center_phase_angle(PLANET).vals + bp.center_scattering_angle(PLANET).vals

    assert total == pytest.approx(PI)


def test_lighting_backplanes_are_cached(bp: Backplane) -> None:
    """A backplane already computed is returned rather than recomputed."""

    assert bp.incidence_angle(PLANET) is bp.incidence_angle(PLANET)
    assert bp.lambert_law(PLANET) is bp.lambert_law(PLANET)

##########################################################################################
