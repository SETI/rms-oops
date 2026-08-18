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
    (from_file's cmatrix/frame_id arguments), plus the generic
    Observation.get_cmatrix() / Observation.set_cmatrix() methods.

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
        time. CASSINI_ISS_<camera>_SPICE is the SpiceFrame wrapping SPICE's
        CASSINI_ISS_<camera>, so its J2000 attitude is exactly
        cspyce.pxform('J2000', 'CASSINI_ISS_<camera>', time). Requires a prior
        plain load (define_camera_frames) so the *_SPICE frame is registered.
        """

        return (Frame.as_wayframe('CASSINI_ISS_' + camera + '_SPICE')
                     .wrt(Frame.J2000).transform_at_time(time).matrix)

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
    def test_default_frames_are_distinct_and_stay_out_of_registry(self):
        """Pin the frame_id=None isolation contract against registry rewrites:
        observations given equal-valued C-matrices must get distinct frame
        objects, and a default set_cmatrix must leave the global frame
        registry and cache unchanged. A registry that dedups or retains
        unregistered frames (e.g. a per-subclass wayframe table) breaks the
        documented per-observation ownership; this test makes that a failure
        here instead of a silent semantic change."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)
        custom = _PERTURBATION * spice0

        # Warm every lazy path once (load + get_cmatrix) before measuring.
        obs1 = from_file(self.filespec, cmatrix=custom)
        _ = obs1.get_cmatrix()

        n_wayframes = len(Frame.WAYFRAME_REGISTRY)
        n_cached = len(Frame.FRAME_CACHE)

        # Equal-valued C-matrices: distinct frame objects per observation.
        obs2 = from_file(self.filespec, cmatrix=custom)
        obs3 = from_file(self.filespec, cmatrix=custom)
        _ = obs2.get_cmatrix()
        _ = obs3.get_cmatrix()

        self.assertIsNot(obs1.frame, obs2.frame)
        self.assertIsNot(obs2.frame, obs3.frame)
        self.assertTrue(np.all(obs2.get_cmatrix().vals == custom.vals))

        # No global retention: the registry and cache did not grow.
        self.assertEqual(len(Frame.WAYFRAME_REGISTRY), n_wayframes)
        self.assertEqual(len(Frame.FRAME_CACHE), n_cached)

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
    def test_reused_frame_id_replaces_primary(self):
        """Re-using a frame_id re-points every observation sharing that
        registered frame; the primary registry definition is cleanly replaced,
        not shadowed by a stale entry (Frame.register's secondary-definition
        path)."""

        baseline = from_file(self.filespec)
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)

        custom1 = _PERTURBATION * spice0
        obs1 = from_file(self.filespec, cmatrix=custom1,
                         frame_id='TEST_REUSED_FRAME')
        self.assertTrue(np.allclose(obs1.get_cmatrix().vals, custom1.vals))

        custom2 = _PERTURBATION * _PERTURBATION * spice0
        obs2 = from_file(self.filespec, cmatrix=custom2,
                         frame_id='TEST_REUSED_FRAME')
        self.assertTrue(np.allclose(obs2.get_cmatrix().vals, custom2.vals))

        # The primary registry definition must reflect the new pointing, not
        # a stale first definition left behind by a secondary registration.
        primary = Frame.as_primary_frame('TEST_REUSED_FRAME')
        attitude = primary.wrt(Frame.J2000).transform_at_time(baseline.tstart).matrix
        recovered = ISS.CMATRIX_ROTATION.transpose() * attitude
        self.assertTrue(np.allclose(recovered.vals, custom2.vals))
        self.assertFalse(np.allclose(recovered.vals, custom1.vals))

    #===========================================================================
    def test_colliding_frame_id_raises(self):
        """A frame_id already registered by anything other than set_cmatrix
        (e.g. the global camera frames or J2000) must raise rather than
        silently replacing that frame's primary definition."""

        baseline = from_file(self.filespec)
        m0 = baseline.get_cmatrix()
        camera = baseline.dict['INSTRUMENT_ID'][3:] + 'C'
        spice0 = self._spice_cmatrix(camera, baseline.tstart)
        custom = _PERTURBATION * spice0

        with self.assertRaises(ValueError):
            from_file(self.filespec, cmatrix=custom,
                      frame_id='CASSINI_ISS_' + camera)

        with self.assertRaises(ValueError):
            from_file(self.filespec, cmatrix=custom, frame_id='J2000')

        # The refused registrations left the global pointing untouched.
        plain = from_file(self.filespec)
        self.assertTrue(np.allclose(plain.get_cmatrix().vals, m0.vals))

    #===========================================================================
    def test_rejects_non_rotation_cmatrix(self):
        """set_cmatrix (and therefore from_file's cmatrix argument) must
        reject anything that is not a proper floating-point rotation: NaNs,
        zeros, scalings, reflections, and boolean/integer data."""

        obs = from_file(self.filespec, cmatrix=oops.Matrix3(np.eye(3)))

        bad_matrices = [
            np.diag([1., 2., 1.]),          # scaling, det = 2
            np.diag([1., 1., -1.]),         # reflection, det = -1
            np.full((3, 3), np.nan),        # non-finite
            np.zeros((3, 3)),               # det = 0
            np.eye(3, dtype=bool),          # boolean data
            np.eye(3, dtype=int),           # integer data
        ]
        for bad in bad_matrices:
            with self.assertRaises(ValueError, msg=repr(bad)):
                obs.set_cmatrix(bad)

        with self.assertRaises(ValueError):
            from_file(self.filespec, cmatrix=np.diag([1., 1., -1.]))

        # A legitimate rotation still passes.
        obs.set_cmatrix(_PERTURBATION)
        self.assertTrue(np.allclose(obs.get_cmatrix().vals, _PERTURBATION.vals))

    #===========================================================================
    def test_frame_id_without_cmatrix_raises(self):
        """frame_id is only meaningful together with cmatrix; passing it alone
        must raise rather than being silently ignored."""

        with self.assertRaises(ValueError):
            from_file(self.filespec, frame_id='TEST_FRAME_ID_ALONE')

    #===========================================================================
    def test_missing_host_contract_raises(self):
        """set_cmatrix/get_cmatrix on an observation with no `host` subfield
        raise an informative ValueError rather than a bare AttributeError."""

        baseline = from_file(self.filespec)
        bare = oops.obs.Snapshot(('v', 'u'), baseline.tstart, baseline.texp,
                                 baseline.fov, path=baseline.path,
                                 frame=baseline.frame)

        self.assertFalse(hasattr(bare, 'host'))

        with self.assertRaises(ValueError):
            bare.get_cmatrix()

        with self.assertRaises(ValueError):
            bare.set_cmatrix(oops.Matrix3(np.eye(3)))

    #===========================================================================
    def test_custom_cmatrix_loads_no_ck(self):
        """A custom cmatrix never loads a CK (no SPICE pointing dependency)."""

        self.assertFalse(np.any(Cassini.CK_LOADED))

        _ = from_file(self.filespec, cmatrix=oops.Matrix3(np.eye(3)))

        self.assertFalse(np.any(Cassini.CK_LOADED))

    #===========================================================================
    def test_spice_pointing_loads_ck(self):
        """SPICE pointing (no cmatrix) loads a CK."""

        self.assertFalse(np.any(Cassini.CK_LOADED))
        _ = from_file(self.filespec)
        self.assertTrue(np.any(Cassini.CK_LOADED))

    #===========================================================================
    def test_custom_cmatrix_reports_no_ck_kernels(self):
        """A custom cmatrix's spice_kernels must not report any CK, even one
        furnished for an unrelated reason (e.g. the gapfill CKs loaded
        unconditionally by Cassini.initialize(), or a CK furnished earlier in
        the session by a plain SPICE-pointed load)."""

        # A plain load furnishes and reports at least one CK.
        plain = from_file(self.filespec)
        self.assertTrue(any(name.endswith('.bc') for name in plain.spice_kernels))

        # A subsequent custom-cmatrix load must report none, despite CKs
        # (including the one just furnished above) being present in the
        # kernel pool.
        custom = from_file(self.filespec, cmatrix=oops.Matrix3(np.eye(3)))
        self.assertFalse(any(name.endswith('.bc') for name in custom.spice_kernels))

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
