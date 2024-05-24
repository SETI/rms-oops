################################################################################
# tests/hosts/hst/__init__.py
################################################################################

class Test_HST(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support import TESTDATA_PARENT_DIRECTORY
        import cspyce
        from .acs.hrc import HRC

        APR = oops.DPR * 3600.

        prefix = os.path.join(TESTDATA_PARENT_DIRECTORY, "hst")
        snapshot = from_file(os.path.join(prefix, "ibht07svq_drz.fits"))
        self.assertEqual(snapshot.instrument, "WFC3")
        self.assertEqual(snapshot.detector, "IR")

        snapshot = from_file(os.path.join(prefix, "ibht07svq_ima.fits"))
        self.assertEqual(snapshot.instrument, "WFC3")
        self.assertEqual(snapshot.detector, "IR")

        snapshot = from_file(os.path.join(prefix, "ibht07svq_raw.fits"))
        self.assertEqual(snapshot.instrument, "WFC3")
        self.assertEqual(snapshot.detector, "IR")

        snapshot = from_file(os.path.join(prefix, "ibu401nnq_flt.fits"))
        self.assertEqual(snapshot.instrument, "WFC3")
        self.assertEqual(snapshot.detector, "UVIS")

        snapshot = from_file(os.path.join(prefix, "j9dh35h7q_raw.fits"))
        self.assertEqual(snapshot.instrument, "ACS")
        self.assertEqual(snapshot.detector, "HRC")

        snapshot = from_file(os.path.join(prefix, "j96o01ioq_raw.fits"))
        self.assertEqual(snapshot.instrument, "ACS")
        self.assertEqual(snapshot.detector, "WFC")

        snapshot = from_file(os.path.join(prefix, "n43h05b3q_raw.fits"))
        self.assertEqual(snapshot.instrument, "NICMOS")
        self.assertEqual(snapshot.detector, "NIC2")

        snapshot = from_file(os.path.join(prefix, "ua1b0309m_d0m.fits"), layer=2)
        self.assertEqual(snapshot.instrument, "WFPC2")
        self.assertEqual(snapshot.detector, "")
        self.assertEqual(snapshot.layer, 2)

        snapshot = from_file(os.path.join(prefix, "ua1b0309m_d0m.fits"), layer=3)
        self.assertEqual(snapshot.instrument, "WFPC2")
        self.assertEqual(snapshot.detector, "")
        self.assertEqual(snapshot.layer, 3)

        self.assertRaises(IOError, from_file, os.path.join(prefix, "ua1b0309m_d0m.fits"),
                                              **{"mask":"required"})

        self.assertRaises(IOError, from_file, os.path.join(prefix, "a.b.c.d"))

        # Raw ACS/HRC, full-frame with overscan pixels
        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY, "hst/j9dh35h7q_raw.fits")
        snapshot = from_file(filespec)
        hst_file = pyfits.open(filespec)
        self.assertEqual(snapshot.filter, "F475W")
        self.assertEqual(snapshot.detector, "HRC")

        # Test time_limits()
        (time0, time1) = HST().time_limits(hst_file)

        self.assertTrue(time1 - time0 - hst_file[0].header["EXPTIME"] > -1.e-8)
        self.assertTrue(time1 - time0 - hst_file[0].header["EXPTIME"] <  1.e-8)

        str0 = cspyce.et2utc(time0, "ISOC", 0)
        self.assertEqual(str0, hst_file[0].header["DATE-OBS"] + "T" +
                               hst_file[0].header["TIME-OBS"])

        # Test get_fov()
        fov = HRC().define_fov(hst_file)
        shape = tuple(fov.uv_shape.vals)
        buffer = np.empty(shape + (2,))
        buffer[:,:,0] = np.arange(shape[0])[..., np.newaxis] + 0.5
        buffer[:,:,1] = np.arange(shape[1]) + 0.5
        pixels = oops.Pair(buffer)

        self.assertTrue(not np.any(fov.uv_is_outside(pixels)))

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

        self.assertTrue(abs(test_pixels - pixels).max() < 1.e-7)

        # Separations between pixels in arcsec are around 0.025
        seps = los[1:].sep(los[:-1])
        self.assertTrue(np.min(seps.vals) * APR > 0.028237 * NSTEP)
        self.assertTrue(np.max(seps.vals) * APR < 0.028648 * NSTEP)

        seps = los[:,1:].sep(los[:,:-1])
        self.assertTrue(np.min(seps.vals) * APR > 0.024547 * NSTEP)
        self.assertTrue(np.max(seps.vals) * APR < 0.025186 * NSTEP)

        # Pixel area factors are near unity
        areas = fov.area_factor(pixels)
        self.assertTrue(np.min(areas.vals) > 1.102193)
        self.assertTrue(np.max(areas.vals) < 1.149735)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
