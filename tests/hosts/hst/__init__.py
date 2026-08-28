################################################################################
# tests/hosts/hst/__init__.py
################################################################################

import pytest


def test_hst():
    from oops.unittester_support import TEST_DATA_PREFIX
    import cspyce
    from .acs.hrc import HRC

    APR = oops.DPR * 3600.

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ibht07svq_drz.fits')
    assert snapshot.instrument == 'WFC3'
    assert snapshot.detector == 'IR'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ibht07svq_ima.fits')
    assert snapshot.instrument == 'WFC3'
    assert snapshot.detector == 'IR'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ibht07svq_raw.fits')
    assert snapshot.instrument == 'WFC3'
    assert snapshot.detector == 'IR'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ibu401nnq_flt.fits')
    assert snapshot.instrument == 'WFC3'
    assert snapshot.detector == 'UVIS'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/j9dh35h7q_raw.fits')
    assert snapshot.instrument == 'ACS'
    assert snapshot.detector == 'HRC'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/j96o01ioq_raw.fits')
    assert snapshot.instrument == 'ACS'
    assert snapshot.detector == 'WFC'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/n43h05b3q_raw.fits')
    assert snapshot.instrument == 'NICMOS'
    assert snapshot.detector == 'NIC2'

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ua1b0309m_d0m.fits', layer=2)
    assert snapshot.instrument == 'WFPC2'
    assert snapshot.detector == ''
    assert snapshot.layer == 2

    snapshot = from_file(TEST_DATA_PREFIX / 'hst/ua1b0309m_d0m.fits', layer=3)
    assert snapshot.instrument == 'WFPC2'
    assert snapshot.detector == ''
    assert snapshot.layer == 3

    with pytest.raises(IOError):
        from_file(TEST_DATA_PREFIX / 'ua1b0309m_d0m.fits', **{'mask':'required'})

    with pytest.raises(IOError):
        from_file(TEST_DATA_PREFIX / 'a.b.c.d')

    # Raw ACS/HRC, full-frame with overscan pixels
    filespec = TEST_DATA_PREFIX / 'hst/j9dh35h7q_raw.fits'
    snapshot = from_file(filespec)
    hst_file = pyfits.open(filespec)
    assert snapshot.filter == 'F475W'
    assert snapshot.detector == 'HRC'

    # Test time_limits()
    (time0, time1) = HST().time_limits(hst_file)

    assert time1 - time0 - hst_file[0].header['EXPTIME'] > -1.e-8
    assert time1 - time0 - hst_file[0].header['EXPTIME'] <  1.e-8

    str0 = cspyce.et2utc(time0, 'ISOC', 0)
    assert (str0
            == hst_file[0].header['DATE-OBS'] + 'T' + hst_file[0].header['TIME-OBS'])

    # Test get_fov()
    fov = HRC().define_fov(hst_file)
    shape = tuple(fov.uv_shape.vals)
    buffer = np.empty(shape + (2,))
    buffer[:,:,0] = np.arange(shape[0])[..., np.newaxis] + 0.5
    buffer[:,:,1] = np.arange(shape[1]) + 0.5
    pixels = oops.Pair(buffer)

    assert not np.any(fov.uv_is_outside(pixels))

    # Confirm that a fov.PolynomialFOV is reversible
    #
    # This is SLOW for a million pixels but it works. I have done a bit of
    # optimization and appear to have reached the point of diminishing
    # returns.
    #
    # los = fov.los_from_uv(pixels)
    # test_pixels = fov.uv_from_los(los)

    # Faster version, 1/64 pixels
    NSTEP = 256
    pixels = oops.Pair(buffer[::NSTEP,::NSTEP])
    los = fov.los_from_uv(pixels)
    test_pixels = fov.uv_from_los(los)

    assert abs(test_pixels - pixels).max() < 1.e-7

    # Separations between pixels in arcsec are around 0.025
    seps = los[1:].sep(los[:-1])
    assert np.min(seps.vals) * APR > 0.028237 * NSTEP
    assert np.max(seps.vals) * APR < 0.028648 * NSTEP

    seps = los[:,1:].sep(los[:,:-1])
    assert np.min(seps.vals) * APR > 0.024547 * NSTEP
    assert np.max(seps.vals) * APR < 0.025186 * NSTEP

    # Pixel area factors are near unity
    areas = fov.area_factor(pixels)
    assert np.min(areas.vals) > 1.102193
    assert np.max(areas.vals) < 1.149735
################################################################################
