##########################################################################################
# tests/surface/test_spice_shape.py
##########################################################################################

import os
import pytest

import cspyce

from oops.frame.frame_        import Frame
from oops.frame.spiceframe    import SpiceFrame
from oops.path.path_          import Path
from oops.path.spicepath      import SpicePath
from oops.surface.spice_shape import spice_shape
from oops.unittester_support  import TEST_SPICE_PREFIX
import oops.spice_support as spice


@pytest.fixture(autouse=True)
def _spice_shape_kernels():
    spice.initialize()
    # A custom path_id only takes effect when a new SpicePath is constructed, so the
    # registry must be clear of any VENUS path left behind by an earlier test
    Path._reset_caches()
    Frame._reset_caches()
    paths = TEST_SPICE_PREFIX.retrieve(["pck00010.tpc",
                                        "de421.bsp"])
    for path in paths:
        cspyce.furnsh(path)

def test_spice_shape():
    _ = SpicePath("VENUS", "SSB", "J2000", path_id="APHRODITE")

    body = spice_shape("VENUS")
    assert Path.as_path(body.origin).path_id == "APHRODITE"
    assert body._req == 6051.8
    assert body._squash_z == 1.
##########################################################################################
