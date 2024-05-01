################################################################################
# tests/surface/test_limb.py
################################################################################

import numpy as np
import unittest

from polymath          import Scalar, Vector3
from oops.surface.limb import Limb


class Test_Limb(unittest.TestCase):

    def runTest(self):

        from oops.frame                    import Frame
        from oops.path                     import Path
        from oops.surface.centricellipsoid import CentricEllipsoid
        from oops.surface.centricspheroid  import CentricSpheroid
        from oops.surface.ellipsoid        import Ellipsoid
        from oops.surface.graphicellipsoid import GraphicEllipsoid
        from oops.surface.graphicspheroid  import GraphicSpheroid
        from oops.surface.spheroid         import Spheroid
        from polymath                      import Matrix3

        np.random.seed(6922)

        REQ  = 60268.
        RMID = 54364.
        RPOL = 50000.

        NPTS = 1000

        ground = Spheroid('SSB', 'J2000', (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[...,0] = -4 *REQ
        los_vals[...,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[...,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)

        # Check (z,clock) conversions
        (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

        self.assertTrue((track2 - track).norm().median() < 1.e-10)
        self.assertTrue(abs(track.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)
        self.assertTrue(abs(track2.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)

        matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
        (x,y,_) = (matrix * ground.normal(track)).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % Scalar.TWOPI - clock).median() < 1.e-10)

        (x,y,_) = (matrix * ground.normal(track2)).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % Scalar.TWOPI - clock).max() < 1.e-12)

        self.assertTrue(abs((cept - track).sep(los)  - Scalar.HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(los) - Scalar.HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track).sep(limb.ground.normal(track))).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(limb.ground.normal(track2))).median() < 1.e-12)

        cept2 = limb.intercept_from_z_clock(z, clock, obs)
        (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

        # Validate solution
        (cept, t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track).unit()
        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        normal2 = cept - track
        sep = (normal2.sep(normal) + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue(abs(sep).max() < 1.e-10)

        # Validate (lon,lat) conversions without z
        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)

        pos = limb.vector3_from_coords((lon,lat))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon).max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2]).max() < 1.e-6)

        clock = np.random.random(NPTS) * Scalar.TWOPI
        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        # Validate (lon,lat) conversions with z
        z = np.random.random(NPTS) * 10000. - 100.
        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon).max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-10)

        clock = np.random.random(NPTS) * Scalar.TWOPI
        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        # Validate clock angles
        track = limb.groundtrack_from_clock(clock, obs)
        clock2 = limb.clock_from_groundtrack(track, obs)
        track2 = limb.groundtrack_from_clock(clock2, obs)

        self.assertTrue((track2 - track).norm().max() < 1.e-6)

        dclock = (clock2 - clock + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(abs(dclock).max() < 1.e-12)

        # Intercept with derivs
        N = 1000
        obs = Vector3(REQ * (0.95 + np.random.rand(N,3)))
        los = Vector3(np.random.randn(N,3))
        mask = obs.dot(los) > 0
        los[mask] = -los[mask]

        obs.insert_deriv('obs', Vector3.IDENTITY)
        los.insert_deriv('los', Vector3.IDENTITY)

        (pos, t, hints, track) = limb.intercept(obs, los, derivs=True, hints=True,
                                                groundtrack=True)

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (pos1, t1, _, track1) = limb.intercept(obs+dobs[i], los, derivs=False,
                                                   guess=t.wod, hints=hints.wod,
                                                   groundtrack=True)

            (pos2, t2, _, track2) = limb.intercept(obs-dobs[i], los, derivs=False,
                                                   guess=t.wod, hints=hints.wod,
                                                   groundtrack=True)
            dpos_dobs = (pos1 - pos2) / (2*eps)
            self.assertTrue(abs(dpos_dobs - pos.d_dobs.vals[...,i]).max() < 1.e-9)

            dt_dobs = (t1 - t2) / (2*eps)
            self.assertTrue(abs(dt_dobs - t.d_dobs.vals[...,i]).max() < 1.e-9)

            dtrack_dobs = (track1 - track2) / (2*eps)
            self.assertTrue(abs(dtrack_dobs - track.d_dobs.vals[...,i]).max() < 1.e-9)

        eps = 1.e-7
        dlos = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (pos1, t1, _, track1) = limb.intercept(obs, los+dlos[i], derivs=False,
                                                   guess=t.wod, hints=hints.wod,
                                                   groundtrack=True)

            (pos2, t2, _, track2) = limb.intercept(obs, los-dlos[i], derivs=False,
                                                   guess=t.wod, hints=hints.wod,
                                                   groundtrack=True)
            dpos_dlos = (pos1 - pos2) / (2*eps)
            scale = dpos_dlos.norm().median()
            self.assertTrue(abs(dpos_dlos - pos.d_dlos.vals[...,i]).max() < scale * 3.e-8)

            dt_dlos = (t1 - t2) / (2*eps)
            scale = dt_dlos.abs().median()
            self.assertTrue(abs(dt_dlos - t.d_dlos.vals[...,i]).max() < scale * 3.e-8)

            dtrack_dlos = (track1 - track2) / (2*eps)
            scale = dtrack_dlos.norm().median()
            self.assertTrue(abs(dtrack_dlos - track.d_dlos.vals[...,i]).max() < scale * 3.e-8)

        # intercept_from_z_clock with derivs
        N = 1000
        z = Scalar(REQ * (0.95 + np.random.rand(N)))
        clock = Scalar(np.random.randn(N)) * Scalar.TWOPI
        obs = Vector3(REQ * (1.95 + np.random.rand(N,3)))

        z.insert_deriv('z', Scalar.ONE)
        clock.insert_deriv('clock', Scalar.ONE)
        obs.insert_deriv('obs', Vector3.IDENTITY)

        (pos, track) = limb.intercept_from_z_clock(z, clock, obs, derivs=True,
                                                   groundtrack=True)
        eps = 1.
        (pos1, track1) = limb.intercept_from_z_clock(z + eps, clock, obs,
                                                     derivs=False,
                                                     groundtrack=True)
        (pos2, track2) = limb.intercept_from_z_clock(z - eps, clock, obs,
                                                     derivs=False,
                                                     groundtrack=True)
        dpos_dz = (pos1 - pos2) / (2*eps)
        self.assertTrue(abs(dpos_dz - pos.d_dz).max() < 1.e-9)

        dtrack_dz = (track1 - track2) / (2*eps)
        self.assertTrue(abs(dtrack_dz - track.d_dz).max() < 1.e-9)

        eps = 1.e-6
        (pos1, track1) = limb.intercept_from_z_clock(z, clock+eps, obs,
                                                     derivs=False,
                                                     groundtrack=True)
        (pos2, track2) = limb.intercept_from_z_clock(z, clock-eps, obs,
                                                     derivs=False,
                                                     groundtrack=True)

        dpos_dclock = (pos1 - pos2) / (2*eps)
        scale = dpos_dclock.norm().median()
        self.assertTrue(abs(dpos_dclock - pos.d_dclock).max() < scale * 3.e-8)

        dtrack_dclock = (track1 - track2) / (2*eps)
        scale = dtrack_dclock.norm().median()
        self.assertTrue(abs(dtrack_dclock - track.d_dclock).max() < scale * 3.e-8)

        eps = 1.
        dobs = ((eps,0,0), (0,eps,0), (0,0,eps))
        for i in range(3):
            (pos1, track1) = limb.intercept_from_z_clock(z, clock, obs+dobs[i],
                                                         derivs=False,
                                                         groundtrack=True)

            (pos2, track2) = limb.intercept_from_z_clock(z, clock, obs-dobs[i],
                                                         derivs=False,
                                                         groundtrack=True)
            dpos_dobs = (pos1 - pos2) / (2*eps)
            scale = dpos_dobs.norm().median()
            self.assertTrue(abs(dpos_dobs - pos.d_dobs.vals[...,i]).max() < scale * 1.e-9)

            dtrack_dobs = (track1 - track2) / (2*eps)
            scale = dtrack_dobs.norm().median()
            self.assertTrue(abs(dtrack_dobs - track.d_dobs.vals[...,i]).max() < scale * 1.e-9)

        ####################

        ground = Ellipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        # Check (z,clock) conversions
        (z, clock, track2) = limb.z_clock_from_intercept(cept, obs, groundtrack=True)

        self.assertTrue((track2 - track).norm().median() < 1.e-10)
        self.assertTrue(abs(track.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)
        self.assertTrue(abs(track2.element_mul(limb.ground.unsquash).norm() -
                            limb.ground.req).median() < 1.e-10)

        matrix = Matrix3.twovec(-obs, 2, Vector3.ZAXIS, 0)
        (x,y,_) = (matrix * normal).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % Scalar.TWOPI - clock).median() < 1.e-10)

        (x,y,_) = (matrix * limb.normal(track2)).to_scalars()
        self.assertTrue(abs(y.arctan2(x) % Scalar.TWOPI - clock).max() < 1.e-12)

        self.assertTrue(abs((cept - track).sep(los)  - Scalar.HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(los) - Scalar.HALFPI).median() < 1.e-12)
        self.assertTrue(abs((cept - track).sep(limb.ground.normal(track))).median() < 1.e-12)
        self.assertTrue(abs((cept - track2).sep(limb.ground.normal(track2))).median() < 1.e-12)

        cept2 = limb.intercept_from_z_clock(z, clock, obs)
        (z2, clock2) = limb.z_clock_from_intercept(cept2, obs)

        # Validate solution
        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        normal2 = cept - track
        sep = (normal2.sep(normal) + Scalar.HALFPI) % Scalar.PI - Scalar.HALFPI
        self.assertTrue(abs(sep).max() < 1.e-10)

        # Validate (lon,lat) conversions
        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        self.assertTrue(abs(coords[0] - lon).max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        clock = np.random.random(NPTS) * Scalar.TWOPI
        obs = Vector3.from_scalars(REQ * np.random.random(NPTS) + 1.5*REQ,
                                   REQ * np.random.random(NPTS),
                                   REQ * np.random.random(NPTS))

        # Validate clock angles
        track = limb.groundtrack_from_clock(clock, obs)
        clock2 = limb.clock_from_groundtrack(track, obs)
        track2 = limb.groundtrack_from_clock(clock2, obs)

        self.assertTrue((track2 - track).norm().max() < 1.e-6)

        dclock = (clock2 - clock + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(abs(dclock).max() < 1.e-12)

        ####################

        ground = CentricSpheroid('SSB', 'J2000', (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = GraphicSpheroid('SSB', 'J2000', (REQ, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = CentricEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        ####################

        ground = GraphicEllipsoid('SSB', 'J2000', (REQ, RMID, RPOL))
        limb = Limb(ground)

        obs = Vector3([4*REQ,0,0])

        los_vals = np.empty((220,220,3))
        los_vals[:,:,0] = -4 *REQ
        los_vals[:,:,1] = np.arange(-1.10,1.10,0.01)[:,np.newaxis] * REQ
        los_vals[:,:,2] = np.arange(-1.10,1.10,0.01) * REQ
        los = Vector3(los_vals)

        (cept,t, track) = limb.intercept(obs, los, groundtrack=True)
        normal = limb.normal(track)

        self.assertTrue(abs(normal.sep(los) - Scalar.HALFPI).max() < 1.e-12)

        lon = np.random.random(NPTS) * Scalar.TWOPI
        lat = np.arcsin(np.random.random(NPTS) * 2. - 1.)
        z = np.random.random(NPTS) * 10000.

        pos = limb.vector3_from_coords((lon,lat,z))
        coords = limb.coords_from_vector3(pos, axes=3)

        diffs = abs(coords[0] - lon)
        diffs = (diffs + Scalar.PI) % Scalar.TWOPI - Scalar.PI
        self.assertTrue(diffs.max() < 1.e-12)
        self.assertTrue(abs(coords[1] - lat).max() < 1.e-12)
        self.assertTrue(abs(coords[2] - z).max() < 1.e-6)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':

    import oops
    oops.config.LOGGING.on('     ')
    unittest.main(verbosity=2)

################################################################################
