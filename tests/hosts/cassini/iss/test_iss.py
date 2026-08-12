################################################################################
# tests/hosts/cassini/iss/test_iss.py
################################################################################

import numpy as np
import unittest

import oops
from oops.body                import Body
from oops.frame               import Frame
from oops.hosts.cassini       import Cassini
from oops.hosts.cassini.iss   import ISS, from_file, CMATRIX_ROTATION
from oops.unittester_support  import TEST_DATA_PREFIX

# A rotation distinguishable from any SPICE-derived pointing, used to build
# fake "custom" C-matrices below.
_PERTURBATION = oops.Matrix3([[0,1,0],[-1,0,0],[0,0,1]])


class Test_Cassini_ISS_Cmatrix(unittest.TestCase):
    """Tests for the custom C-matrix support in hosts/cassini/iss.py
    (from_file's cmatrix/frame_id/map_other_camera arguments and
    ISS.set_cmatrix), plus the generic Observation.get_cmatrix() getter.

    The `cmatrix` accepted by from_file is the SPICE camera-frame C-matrix (the
    J2000 -> CASSINI_ISS_<camera> rotation, as returned by cspyce.pxform). The
    oops observation frame is that frame rotated 180 degrees about the boresight;
    get_cmatrix() returns the pointing back in that SPICE C-matrix convention.
    """

    FILESPEC = 'cassini/ISS/W1573721822_1.IMG'      # a WAC image

    def setUp(self):
        # A mapped custom C-matrix can override the global CASSINI_ISS_NAC/WAC
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
    def _spice_cmatrix(self, camera, time):
        """The recorded SPICE camera-frame C-matrix for the given camera and
        time. CASSINI_ISS_<camera>_FLIPPED is the SpiceFrame wrapping SPICE's
        CASSINI_ISS_<camera>, so its J2000 attitude is exactly
        cspyce.pxform('J2000', 'CASSINI_ISS_<camera>', time). Requires a prior
        plain load (define_camera_frames) so the *_FLIPPED frame is registered.
        """

        return (Frame.as_wayframe('CASSINI_ISS_' + camera + '_FLIPPED')
                     .wrt(Frame.J2000).transform_at_time(time).matrix)

    #===========================================================================
    def test_convention_conversions(self):
        """ISS.oops_from_spice / spice_from_oops convert between the two
        conventions and are mutual inverses."""

        # Mutual inverses, for an arbitrary rotation.
        m = _PERTURBATION * CMATRIX_ROTATION * _PERTURBATION
        self.assertTrue(np.allclose(m.vals,
                        ISS.spice_from_oops(ISS.oops_from_spice(m)).vals))
        self.assertTrue(np.allclose(m.vals,
                        ISS.oops_from_spice(ISS.spice_from_oops(m)).vals))

    #===========================================================================
    def test_get_cmatrix_inverts_set(self):
        """get_cmatrix returns the SPICE-convention C-matrix (matching
        pxform) and inverts the cmatrix given to from_file."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        # For a SPICE-loaded observation, get_cmatrix matches pxform.
        self.assertTrue(np.allclose(
                baseline.get_cmatrix(time=baseline.tstart).vals, spice0.vals))

        # It recovers the exact cmatrix given to from_file (the custom Snapshot
        # frame is fixed in time, so no time argument is needed).
        custom = _PERTURBATION * spice0
        obs = from_file(self.filespec, cmatrix=custom)
        recovered = obs.get_cmatrix()
        self.assertTrue(np.allclose(recovered.vals, custom.vals))

        # Round-trip: feeding it back into from_file reproduces the pointing.
        obs2 = from_file(self.filespec, cmatrix=recovered)
        self.assertTrue(np.allclose(obs2.get_cmatrix().vals,
                                    obs.get_cmatrix().vals))

    #===========================================================================
    def test_default_frame_is_unregistered_and_isolated(self):
        """With frame_id=None each observation gets its own unregistered frame,
        so loading another image never changes an earlier one's pointing."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        custom1 = _PERTURBATION * spice0
        obs1 = from_file(self.filespec, cmatrix=custom1)
        saved1 = obs1.get_cmatrix()

        # The observation owns an unregistered frame, not the global wayframe.
        self.assertFalse(obs1.frame.is_registered())
        self.assertTrue(np.allclose(obs1.get_cmatrix().vals, custom1.vals))

        # Loading a second custom image must not disturb the first.
        custom2 = _PERTURBATION * _PERTURBATION * spice0
        obs2 = from_file(self.filespec, cmatrix=custom2)

        self.assertIsNot(obs1.frame, obs2.frame)
        self.assertTrue(np.all(obs1.get_cmatrix().vals == saved1.vals))
        self.assertTrue(np.allclose(obs2.get_cmatrix().vals, custom2.vals))
        self.assertFalse(np.allclose(obs1.get_cmatrix().vals,
                                     obs2.get_cmatrix().vals))

    #===========================================================================
    def test_default_does_not_leak_into_plain_load(self):
        """A default custom load never touches the global camera frame, so a
        subsequent plain load reads back its own SPICE pointing."""

        baseline = from_file(self.filespec)
        m0 = baseline.get_cmatrix()
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        _ = from_file(self.filespec, cmatrix=_PERTURBATION * spice0)

        plain = from_file(self.filespec)
        self.assertTrue(np.allclose(plain.get_cmatrix().vals, m0.vals))

    #===========================================================================
    def test_explicit_frame_id_leaves_global_untouched(self):
        """A custom cmatrix with an explicit frame_id registers a separate
        frame (with the boundary rotation applied) under that frame ID."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)
        custom = _PERTURBATION * spice0

        obs = from_file(self.filespec, cmatrix=custom,
                        frame_id='TEST_ISS_CUSTOM_FRAME')

        self.assertEqual(obs.frame.frame_id, 'TEST_ISS_CUSTOM_FRAME')
        self.assertTrue(np.allclose(obs.get_cmatrix().vals, custom.vals))

        # The global camera frame is untouched.
        unaffected = from_file(self.filespec)
        self.assertTrue(np.allclose(unaffected.get_cmatrix().vals,
                                    baseline.get_cmatrix().vals))

    #===========================================================================
    def test_map_other_camera(self):
        """map_other_camera=True derives the co-mounted camera's frame from
        the fixed, SPICE-derived inter-camera rotation, preserving it under a
        custom C-matrix."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        other = 'NAC' if camera == 'WAC' else 'WAC'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        baseline_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                             .transform_at_time(baseline.tstart).matrix)

        custom = _PERTURBATION * spice0
        obs = from_file(self.filespec, cmatrix=custom, map_other_camera=True)

        self.assertEqual(obs.frame.frame_id, 'CASSINI_ISS_' + camera)

        # The fixed inter-camera rotation is preserved under the override.
        new_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                        .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                        .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(new_rel.vals, baseline_rel.vals))

        # ...and the other camera's absolute pointing was updated accordingly.
        other_matrix = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.J2000)
                             .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(other_matrix.vals,
                                    (baseline_rel * CMATRIX_ROTATION * custom).vals))

    #===========================================================================
    def test_map_reclaims_global_on_plain_load(self):
        """A mapped custom load overrides the global camera frames; a later
        plain load must rebuild them from SPICE rather than inheriting the
        custom pointing."""

        baseline = from_file(self.filespec)
        m0 = baseline.get_cmatrix()
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        _ = from_file(self.filespec, cmatrix=_PERTURBATION * spice0,
                      map_other_camera=True)

        plain = from_file(self.filespec)
        self.assertTrue(np.allclose(plain.get_cmatrix().vals, m0.vals))

    #===========================================================================
    def test_map_after_map_override(self):
        """A mapped load followed by another mapped load must still derive the
        inter-camera rotation from the dedicated *_SPICE frames, not from the
        CASSINI_ISS_<camera> frames a prior mapped load overrode."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        other = 'NAC' if camera == 'WAC' else 'WAC'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        # The true, SPICE-derived inter-camera rotation.
        baseline_rel = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.as_wayframe('CASSINI_ISS_' + camera))
                             .transform_at_time(baseline.tstart).matrix)

        # First mapped load with an unrelated pointing overrides both global
        # camera frames.
        bogus = _PERTURBATION * _PERTURBATION * spice0
        _ = from_file(self.filespec, cmatrix=bogus, map_other_camera=True)

        # Second mapped load: rel must come from the dedicated *_SPICE frames,
        # unaffected by the override above.
        custom = _PERTURBATION * spice0
        _ = from_file(self.filespec, cmatrix=custom, map_other_camera=True)

        other_matrix = (Frame.as_wayframe('CASSINI_ISS_' + other)
                             .wrt(Frame.J2000)
                             .transform_at_time(baseline.tstart).matrix)
        self.assertTrue(np.allclose(other_matrix.vals,
                                    (baseline_rel * CMATRIX_ROTATION * custom).vals))

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
        m0 = baseline.get_cmatrix()
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        _ = from_file(self.filespec, cmatrix=_PERTURBATION * spice0,
                      map_other_camera=True)

        ISS.reset()
        restored = from_file(self.filespec)
        self.assertTrue(np.allclose(restored.get_cmatrix().vals, m0.vals))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
