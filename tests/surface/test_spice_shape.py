##########################################################################################
# tests/surface/test_spice_shape.py
##########################################################################################

import pytest

import cspyce

from oops.frame.frame_        import Frame
from oops.path.path_          import Path
from oops.path.spicepath      import SpicePath
from oops.surface.ellipsoid   import Ellipsoid
from oops.surface.spheroid    import Spheroid
from oops.surface.spice_shape import spice_shape
from programs.gold_master.test_support  import TEST_SPICE_PREFIX
import oops.spice_support as spice


@pytest.fixture(autouse=True)
def _spice_shape_kernels():
    spice.initialize()
    # A custom path_id only takes effect when a new SpicePath is constructed, so the
    # registry must be clear of any VENUS path left behind by an earlier test
    Path._reset_caches()
    Frame._reset_caches()
    paths = TEST_SPICE_PREFIX.retrieve(["naif0009.tls",
                                        "pck00010.tpc",
                                        "de421.bsp"])
    for path in paths:
        cspyce.furnsh(path)

def test_spice_shape():
    _ = SpicePath("VENUS", "SSB", "J2000", path_id="APHRODITE")

    body = spice_shape("VENUS")
    assert Path.as_path(body.origin).path_id == "APHRODITE"
    assert body._req == 6051.8
    assert body._squash_z == 1.

def test_a_triaxial_body_becomes_an_ellipsoid() -> None:
    """A body whose equatorial radii differ needs all three axes."""

    body = spice_shape('PHOBOS')

    assert isinstance(body, Ellipsoid)
    assert body.radii[0] != body.radii[1]


def test_a_body_of_revolution_becomes_a_spheroid() -> None:
    """Equal equatorial radii leave only two distinct axes."""

    body = spice_shape('VENUS')

    assert isinstance(body, Spheroid)
    assert body.radii[0] == body.radii[1]


def test_the_frame_can_be_given_rather_than_inferred() -> None:
    """A frame handed in replaces the one the SPICE body code would select."""

    body = spice_shape('VENUS', frame=Frame.J2000)

    assert body.frame is Frame.J2000


def test_default_radii_stand_in_for_a_body_the_kernels_do_not_describe() -> None:
    """A body with no RADII in the kernel pool falls back on the values supplied."""

    body = spice_shape('CASSINI', frame=Frame.J2000, default_radii=(2., 2., 1.))

    assert isinstance(body, Spheroid)
    assert body._req == 2.


def test_a_body_with_no_radii_and_no_default_is_refused() -> None:
    """Without radii from either source there is no shape to build."""

    with pytest.raises(KeyError, match='radii are not available'):
        spice_shape('CASSINI', frame=Frame.J2000)

##########################################################################################
