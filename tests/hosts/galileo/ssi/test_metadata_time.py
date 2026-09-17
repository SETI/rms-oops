##########################################################################################
# tests/hosts/galileo/ssi/test_metadata_time.py
##########################################################################################
"""Timing of Galileo SSI images whose label gives no IMAGE_TIME.

Some RAW_CAL frames in GO_0002 and GO_0003 have IMAGE_TIME = UNK. Their times must
come from SPACECRAFT_CLOCK_START_COUNT, never from a placeholder value.
"""

import cspyce
import julian
import pytest

import oops.hosts.galileo.ssi as ssi
from programs.gold_master.test_support import TEST_DATA_PREFIX

# GO_0002/RAW_CAL/C0003061200R.LBL: a post-launch checkout frame with no IMAGE_TIME
UNKNOWN_TIME_LABEL: dict = {
    'SPACECRAFT_CLOCK_START_COUNT': '00030612.00',
    'IMAGE_TIME': 'UNK',
    'EXPOSURE_DURATION': 800.0,
    'FILTER_NAME': 'CLEAR',
    'TARGET_NAME': 'BLACK_SKY',
    'TELEMETRY_FORMAT_ID': 'HIM',
}

# GO_0002/VENUS/C0018062600R.LBL: a frame that gives both fields
KNOWN_TIME_LABEL: dict = {
    'SPACECRAFT_CLOCK_START_COUNT': '00180626.00',
    'IMAGE_TIME': '1990-02-10T05:12:17.082Z',
    'EXPOSURE_DURATION': 800.0,
    'FILTER_NAME': 'CLEAR',
    'TARGET_NAME': 'VENUS',
    'TELEMETRY_FORMAT_ID': 'IM4',
}

# The same checkout frame in the shared test-data tree, for the from_file path
UNKNOWN_TIME_IMAGE = 'galileo/GO_0002/RAW_CAL/C0003061200R.IMG'

GALILEO_LAUNCH_TDB = julian.tdb_from_tai(julian.tai_from_iso('1989-10-18'))


@pytest.fixture(scope='module', autouse=True)
def _initialize_ssi() -> None:
    """Load the SSI instrument and SCLK kernels; skip if SPICE is unavailable."""

    try:
        ssi.initialize()
    except (FileNotFoundError, OSError, RuntimeError) as e:
        pytest.skip('Galileo SPICE kernels unavailable: ' + str(e))


def test_unknown_image_time_comes_from_sclk() -> None:
    """With IMAGE_TIME = UNK, the times follow the spacecraft clock count."""

    meta = ssi.Metadata(dict(UNKNOWN_TIME_LABEL))

    assert meta.time_from_sclk
    assert meta.tstart == cspyce.scs2e(-77, '30612:0:0:0')
    assert meta.tstop == pytest.approx(meta.tstart + 0.8)

    # Nine days after launch, not the J2000 epoch
    assert meta.tstart > GALILEO_LAUNCH_TDB
    iso = julian.iso_from_tai(julian.tai_from_tdb(meta.tstart), digits=0)
    assert iso.startswith('1989-10-27T21:11:')


def test_known_image_time_is_untouched() -> None:
    """A label with IMAGE_TIME keeps it, and the clock count agrees to seconds."""

    meta = ssi.Metadata(dict(KNOWN_TIME_LABEL))

    assert not meta.time_from_sclk
    expected = julian.tdb_from_tai(julian.tai_from_iso(KNOWN_TIME_LABEL['IMAGE_TIME']))
    assert meta.tstart == expected
    assert meta.tstop == pytest.approx(expected + 0.8)

    # The conversion used for unknown times lands within seconds of IMAGE_TIME
    from_sclk = ssi.Metadata.time_from_sclk_count(
                                    KNOWN_TIME_LABEL['SPACECRAFT_CLOCK_START_COUNT'])
    assert abs(from_sclk - expected) < 10.


def test_from_file_carries_sclk_time_onto_snapshot() -> None:
    """from_file forwards the derived time and the time_from_sclk flag to the Snapshot."""

    path = TEST_DATA_PREFIX / UNKNOWN_TIME_IMAGE
    try:
        path.retrieve()
        path.with_suffix('.LBL').retrieve()
        obs = ssi.from_file(path)
    except (FileNotFoundError, OSError) as e:
        pytest.skip('Galileo test data unavailable: ' + str(e))

    assert obs.time_from_sclk
    assert obs.texp == pytest.approx(0.8)
    assert obs.time[0] == cspyce.scs2e(-77, '30612:0:0:0')
    iso = julian.iso_from_tai(julian.tai_from_tdb(obs.time[0]), digits=0)
    assert iso.startswith('1989-10-27T21:11:')

##########################################################################################
