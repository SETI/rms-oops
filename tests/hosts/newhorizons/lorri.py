################################################################################
# tests/hosts/newhorizons/lorri.py
################################################################################

import unittest

class Test_NewHorizons_LORRI(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support import TEST_DATA_PREFIX
        import cspyce

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT",
                             astrometry=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT",
                             data=False, calibration=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT",
                             data=True, calibration=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             headers=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"))
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        self.assertTrue(snapshot.data.shape == (1024,1024))
        self.assertTrue(snapshot.quality.shape == (1024,1024))
        self.assertTrue(snapshot.error.shape == (1024,1024))

        self.assertAlmostEqual(snapshot.time[1]-snapshot.time[0],
                               snapshot.texp)
        self.assertAlmostEqual(snapshot.time[0]+snapshot.texp/2,
                               cspyce.utc2et(snapshot.headers[0]["SPCUTCID"]),
                               places=3)
        self.assertEqual(snapshot.target, "EUROPA")

        fov_1024 = snapshot.fov

        for geom, pointing, offset in [('spice', 'fits90', (-49,-28)),
                                       ('fits', 'spice', (-4,-12)),
                                       ('fits', 'fits90', (-48,-27))]:
            snapshot_fits = from_file(TEST_DATA_PREFIX /
                                      "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                            geom=geom, pointing=pointing, fast_distortion=True)

            self.assertEqual(snapshot.time, snapshot_fits.time)
            self.assertEqual(snapshot.texp, snapshot_fits.texp)

            meshgrid = oops.Meshgrid.for_fov(snapshot.fov, (0,0), limit=(0,0))
            bp = oops.Backplane(snapshot, meshgrid=meshgrid)
            bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
            ra =          bp.right_ascension().vals.astype('float')
            ra_fits =     bp_fits.right_ascension().vals.astype('float')
            dec =         bp.declination().vals.astype('float')
            dec_fits =    bp_fits.declination().vals.astype('float')
            europa =      bp.where_intercepted("europa").vals
            europa_fits = bp_fits.where_intercepted("europa").vals

            self.assertAlmostEqual(ra, ra_fits, places=2)
            self.assertAlmostEqual(dec, dec_fits, places=2)
            self.assertEqual(europa, 0.0)
            self.assertEqual(europa_fits, 0.0)

            # Adjust offset as SPICE kernels change
            orig_fov = snapshot.fov
            orig_fits_fov = snapshot_fits.fov
            snapshot.fov = oops.fov.OffsetFOV(orig_fov, (-4,-13))
            snapshot_fits.fov = oops.fov.OffsetFOV(orig_fits_fov, offset)

            europa_uv = (500,440)
            meshgrid = oops.Meshgrid.for_fov(snapshot.fov, europa_uv,
                                             limit=europa_uv, swap=True)
            meshgrid_fits = oops.Meshgrid.for_fov(snapshot_fits.fov, europa_uv,
                                                  limit=europa_uv, swap=True)
            bp = oops.Backplane(snapshot, meshgrid=meshgrid)
            bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
            long =        bp.longitude("europa").vals.astype('float')
            long_fits =   bp_fits.longitude("europa").vals.astype('float')
            lat =         bp.latitude("europa").vals.astype('float')
            lat_fits =    bp_fits.latitude("europa").vals.astype('float')
            europa =      bp.where_intercepted("europa").vals
            europa_fits = bp_fits.where_intercepted("europa").vals
            snapshot.fov = orig_fov
            snapshot_fits.fov = orig_fits_fov

#             self.assertAlmostEqual(long, long_fits, places=1)
#             self.assertAlmostEqual(lat, lat_fits, places=1)
            self.assertEqual(europa, True)
            self.assertEqual(europa_fits, True)

        europa_ext_iof = (snapshot.extended_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606])).vals
        europa_pt_iof = (snapshot.point_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606],
                                        (440,606))).vals
        self.assertGreater(europa_ext_iof, 0.35)
        self.assertLess(europa_ext_iof, 0.6)
        self.assertAlmostEqual(europa_ext_iof, europa_pt_iof, 1)

        snapshot = from_file(TEST_DATA_PREFIX /
                             "nh/LORRI/LOR_0030710290_0x633_SCI_1.FIT",
                             calibration=False)
        self.assertTrue(snapshot.data.shape == (256,256))
        self.assertTrue(snapshot.quality.shape == (256,256))
        self.assertTrue(snapshot.error.shape == (256,256))

        fov_256 = snapshot.fov

        self.assertAlmostEqual(fov_256.uv_scale.vals[0]/4,
                               fov_1024.uv_scale.vals[0])
        self.assertAlmostEqual(fov_256.uv_scale.vals[1]/4,
                               fov_1024.uv_scale.vals[1])

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
