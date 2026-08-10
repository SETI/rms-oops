################################################################################
# tests/hosts/cassini/iss/test_iss.py
################################################################################

import numpy as np
import unittest

import oops
from oops.body                import Body
from oops.frame               import Frame
from oops.hosts.cassini       import Cassini
from oops.hosts.cassini.iss   import ISS, from_file
from oops.unittester_support  import TEST_DATA_PREFIX

# A rotation distinguishable from any SPICE-derived pointing, used to build
# fake "custom" C-matrices below.
_PERTURBATION = oops.Matrix3([[0,1,0],[-1,0,0],[0,0,1]])


class Test_Cassini_ISS_Cmatrix(unittest.TestCase):
    """Tests for the custom C-matrix support in hosts/cassini/iss.py
    (from_file's cmatrix/frame_id/map_other_camera arguments and
    ISS.set_cmatrix), plus the generic Observation.cmatrix() getter.
    """

    FILESPEC = 'cassini/ISS/W1573721822_1.IMG'      # a WAC image

    def setUp(self):
        # Custom C-matrices can override the global CASSINI_ISS_NAC/WAC
        # frames, so every test starts from a clean registry.
        Body.reset_registry()
        Body.define_solar_system()
        ISS.reset()

        self.filespec = TEST_DATA_PREFIX.retrieve(self.FILESPEC)

    def tearDown(self):
        Body.reset_registry()
        Body.define_solar_system()
        ISS.reset()

    #===========================================================================
    def test_default_override(self):
        """A custom cmatrix with no frame_id overrides the global camera
        frame, and Observation.cmatrix() round-trips it."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        custom = _PERTURBATION * baseline.cmatrix()

        obs = from_file(self.filespec, cmatrix=custom)

        self.assertEqual(obs.frame.frame_id, 'CASSINI_ISS_' + camera)
        self.assertTrue(np.all(obs.cmatrix().vals == custom.vals))

        # The global frame really was replaced, not merely shadowed
        direct = (Frame.as_wayframe('CASSINI_ISS_' + camera)
                       .wrt(Frame.J2000).transform_at_time(obs.tstart).matrix)
        self.assertTrue(np.all(direct.vals == custom.vals))

    #===========================================================================
    def test_explicit_frame_id_leaves_global_untouched(self):
        """A custom cmatrix with an explicit frame_id registers a separate
        frame and leaves the global CASSINI_ISS_<camera> frame alone."""

        baseline = from_file(self.filespec)
        m0 = baseline.cmatrix()
        custom = _PERTURBATION * m0

        obs = from_file(self.filespec, cmatrix=custom,
                        frame_id='TEST_ISS_CUSTOM_FRAME')

        self.assertEqual(obs.frame.frame_id, 'TEST_ISS_CUSTOM_FRAME')
        self.assertTrue(np.all(obs.cmatrix().vals == custom.vals))

        # The global camera frame is untouched
        unaffected = from_file(self.filespec)
        self.assertTrue(np.all(unaffected.cmatrix().vals == m0.vals))

    #===========================================================================
    def test_map_other_camera(self):
        """map_other_camera=True derives the co-mounted camera's frame from
        the fixed, SPICE-derived inter-camera rotation, preserving it under a
        custom C-matrix."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        other = 'NAC' if camera == 'WAC' else 'WAC'
        m0 = baseline.cmatrix()

        baseline_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                             .transform_at_time(baseline.tstart).matrix)

        custom = _PERTURBATION * m0
        obs = from_file(self.filespec, cmatrix=custom, map_other_camera=True)

        self.assertEqual(obs.frame.frame_id, 'CASSINI_ISS_' + camera)
        self.assertTrue(np.all(obs.cmatrix().vals == custom.vals))

        # The fixed inter-camera rotation is preserved under the override
        new_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                        .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                        .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(new_rel.vals, baseline_rel.vals))

        # ...and the other camera's absolute pointing was updated accordingly
        other_matrix = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.J2000)
                             .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(other_matrix.vals,
                                    (baseline_rel * custom).vals))

    #===========================================================================
    def test_map_after_single_camera_override(self):
        """A single-camera override followed by a mapped-camera load must still
        derive the inter-camera rotation from SPICE, not from the overridden
        CASSINI_ISS_<camera> frame."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        other = 'NAC' if camera == 'WAC' else 'WAC'
        m0 = baseline.cmatrix()

        # The true, SPICE-derived inter-camera rotation
        baseline_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                             .transform_at_time(baseline.tstart).matrix)

        # First override the label camera alone with an unrelated custom
        # pointing. This replaces the global CASSINI_ISS_<camera> frame.
        bogus = _PERTURBATION * _PERTURBATION * m0
        _ = from_file(self.filespec, cmatrix=bogus)

        # Now load with mapping. rel must come from the dedicated *_SPICE
        # frames, unaffected by the override above.
        custom = _PERTURBATION * m0
        obs = from_file(self.filespec, cmatrix=custom, map_other_camera=True)

        self.assertTrue(np.all(obs.cmatrix().vals == custom.vals))

        other_matrix = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.J2000)
                             .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(other_matrix.vals,
                                    (baseline_rel * custom).vals))

    #===========================================================================
    def test_no_ck_loaded_without_mapping(self):
        """A custom cmatrix without map_other_camera never loads a CK (no
        SPICE pointing dependency)."""

        self.assertFalse(np.any(Cassini.CK_LOADED))

        _ = from_file(self.filespec, cmatrix=oops.Matrix3(np.eye(3)))

        self.assertFalse(np.any(Cassini.CK_LOADED))

    #===========================================================================
    def test_mapping_or_spice_pointing_loads_ck(self):
        """SPICE pointing (no cmatrix), and map_other_camera=True, each load
        a CK."""

        self.assertFalse(np.any(Cassini.CK_LOADED))
        _ = from_file(self.filespec)
        self.assertTrue(np.any(Cassini.CK_LOADED))

        ISS.reset()
        self.assertFalse(np.any(Cassini.CK_LOADED))
        _ = from_file(self.filespec, cmatrix=oops.Matrix3(np.eye(3)),
                     map_other_camera=True)
        self.assertTrue(np.any(Cassini.CK_LOADED))

    #===========================================================================
    def test_reset_restores_spice_pointing(self):
        """After ISS.reset(), from_file() without cmatrix restores the
        SPICE-derived pointing even if a custom cmatrix was used earlier."""

        baseline = from_file(self.filespec)
        m0 = baseline.cmatrix()

        custom = _PERTURBATION * m0
        _ = from_file(self.filespec, cmatrix=custom)

        ISS.reset()
        restored = from_file(self.filespec)
        self.assertTrue(np.allclose(restored.cmatrix().vals, m0.vals))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
