##########################################################################################
# tests/hosts/galileo/ssi/test_small_satellite_limbs.py
##########################################################################################

import numpy as np
import pytest

import oops
import oops.hosts.galileo.ssi as ssi
from oops.config import LOGGING, SURFACE_PHOTONS
from programs.gold_master.test_support import TEST_DATA_PREFIX

# Frames in which the body fills a few pixels, so most lines of sight have a limb
# point far from it, plus a frame of Jupiter as a control
FRAMES = [
    ('galileo/GO_0019/C9/SML_SATS/C0401639100R.IMG', 'METIS'),
    ('galileo/GO_0019/G8/SML_SATS/C0394682800R.IMG', 'METIS'),
    ('galileo/GO_0019/C9/SML_SATS/C0401703400R.IMG', 'AMALTHEA'),
    ('galileo/GO_0019/G8/JUPITER/C0394455400R.IMG', 'JUPITER'),
]

BODY_COLUMNS = ('distance', 'latitude', 'longitude', 'incidence_angle', 'emission_angle',
                'phase_angle')


def _backplane(path: str) -> oops.Backplane:
    """A Backplane on an undersampled grid over the given image."""

    obs = ssi.from_file(TEST_DATA_PREFIX / path)
    return oops.Backplane(obs, meshgrid=obs.meshgrid(undersample=8))


@pytest.mark.parametrize(('path', 'body'), FRAMES, ids=[f[1] + '-' + f[0][-16:-4]
                                                        for f in FRAMES])
def test_limb_altitudes_do_not_depend_on_evaluation_order(path: str, body: str) -> None:
    """The limb altitudes are the same whether or not the other columns come first."""

    limb_first = _backplane(path)
    altitude_first = limb_first.limb_altitude(body)
    for column in BODY_COLUMNS:
        getattr(limb_first, column)(body)

    columns_first = _backplane(path)
    for column in BODY_COLUMNS:
        getattr(columns_first, column)(body)
    columns_first.limb_clock_angle(body)
    altitude_last = columns_first.limb_altitude(body)

    assert np.array_equal(altitude_last.mask, altitude_first.mask)
    assert abs(altitude_last - altitude_first).max() < SURFACE_PHOTONS.km_precision


@pytest.mark.parametrize(('path', 'body'), FRAMES, ids=[f[1] + '-' + f[0][-16:-4]
                                                        for f in FRAMES])
def test_every_line_of_sight_has_a_limb_point(path: str, body: str) -> None:
    """Rays far from a small body still resolve, and none holds an unphysical value."""

    backplane = _backplane(path)
    LOGGING.reset()
    altitude = backplane.limb_altitude(body)

    radius = oops.Body.lookup(body).surface.radii.max()
    masked = np.count_nonzero(altitude.mask)

    assert masked < 0.005 * altitude.size
    assert altitude.min() > -radius
    assert LOGGING.warnings <= (1 if masked else 0)

##########################################################################################
