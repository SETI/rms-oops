################################################################################
# oops/gold_master/ring.py: Ring backplanes
################################################################################

import numpy as np

from polymath         import Scalar
from oops.constants   import DPR
from oops.gold_master import register_test_suite

def ring_test_suite(bpt):

    bp = bpt.backplane
    for (planet, name) in bpt.planet_ring_pairs:

        # Radius and resolution
        bpt.gmtest(bp.ring_radius(name),
                   name + ' radius (km)',
                   limit=0.1, radius=1)

        bpt.gmtest(bp.ring_radius(name) * bp.ring_angular_resolution(name),
                   name + ' angular resolution (km)',
                   limit=0.1, radius=1.5)

        bpt.gmtest(bp.ring_angular_resolution(name),
                   name + ' angular resolution (deg)',
                   method='degrees', limit=0.01, radius=1.5)

        # Longitude
        bpt.gmtest(bp.ring_longitude(name, reference='aries'),
                   name + ' longitude wrt Aries (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(bp.ring_longitude(name, reference='node'),
                   name + ' longitude wrt node (deg)',
                   method='mod360', limit=0.01, radius=1)

        longitude = bp.ring_longitude(name, reference='obs')
        bpt.gmtest(longitude,
                   name + ' longitude wrt observer (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(longitude - bp.ring_longitude(name, reference='oha'),
                    Scalar.PI,
                    name + ' longitude wrt observer minus wrt OHA (deg)',
                    method='mod360', limit=0.01)

        longitude = bp.ring_longitude(name, reference='sun')
        bpt.gmtest(longitude,
                   name + ' longitude wrt Sun (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(longitude - bp.ring_longitude(name, reference='sha'),
                    Scalar.PI,
                    name + ' longitude wrt Sun minus wrt SHA (deg)',
                    method='mod360', limit=1.e-13)

        # Azimuth
        apparent = bp.ring_azimuth(name, direction='obs', apparent=True)
        actual   = bp.ring_azimuth(name, direction='obs', apparent=False)
        bpt.gmtest(apparent,
                   name + ' azimuth to observer, apparent (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(actual,
                   name + ' azimuth to observer, actual (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(apparent - actual,
                    0.,
                    name + ' azimuth to observer, apparent minus actual (deg)',
                    method='mod360', limit=0.1)

        apparent = bp.ring_azimuth(name, direction='sun', apparent=True)
        actual   = bp.ring_azimuth(name, direction='sun', apparent=False)
        bpt.gmtest(apparent,
                   name + ' azimuth of Sun, apparent (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(actual,
                   name + ' azimuth of Sun, actual (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(apparent - actual,
                    0.,
                    name + ' azimuth of Sun, apparent minus actual (deg)',
                    method='mod360', limit=0.1)

        # Elevation
        apparent = bp.ring_elevation(name, direction='obs', apparent=True)
        actual   = bp.ring_elevation(name, direction='obs', apparent=False)
        bpt.gmtest(apparent,
                   name + ' elevation to observer, apparent (deg)',
                   method='degrees', limit=0.01, radius=1)
        bpt.gmtest(actual,
                   name + ' elevation to observer, actual (deg)',
                   method='degrees', limit=0.01, radius=1)
        bpt.compare(apparent - actual,
                    0.,
                    name + ' elevation to observer, apparent minus actual (deg)',
                    method='degrees', limit=0.1)

        apparent = bp.ring_elevation(name, direction='sun', apparent=True)
        actual   = bp.ring_elevation(name, direction='sun', apparent=False)
        bpt.gmtest(apparent,
                   name + ' elevation of Sun, apparent (deg)',
                   method='degrees', limit=0.01, radius=1)
        bpt.gmtest(actual,
                   name + ' elevation of Sun, actual (deg)',
                   method='degrees', limit=0.01, radius=1)
        bpt.compare(apparent - actual,
                    0.,
                    name + ' elevation of Sun, apparent minus actual (deg)',
                    method='degrees', limit=0.1)

        # Longitude & azimuth tests
        longitude = bp.ring_longitude(name, reference='obs')
        azimuth = bp.ring_azimuth(name, direction='obs')
        bpt.gmtest(azimuth - longitude,
                   name + ' azimuth minus longitude wrt observer (deg)',
                   method='mod360', limit=0.01, radius=1)

        longitude = bp.ring_longitude(name, reference='sun')
        azimuth = bp.ring_azimuth(name, direction='sun')
        bpt.compare(azimuth - longitude, 0.,
                    name + ' azimuth minus longitude wrt Sun (deg)',
                    method='mod360', limit=1.)

        # Sub-observer longitude
        bpt.gmtest(bp.ring_sub_observer_longitude(name, reference='aries'),
                   name + ' sub-observer longitude wrt Aries (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(bp.ring_sub_observer_longitude(name, reference='node'),
                   name + ' sub-observer longitude wrt node (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(bp.ring_sub_observer_longitude(name, reference='sun'),
                   name + ' sub-observer longitude wrt Sun (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(bp.ring_sub_observer_longitude(name, reference='obs'),
                    0.,
                    name + ' sub-observer longitude wrt observer (deg)',
                    method='mod360')

        # Sub-solar longitude
        bpt.gmtest(bp.ring_sub_solar_longitude(name, reference='aries'),
                   name + ' sub-solar longitude wrt Aries (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(bp.ring_sub_solar_longitude(name, reference='node'),
                   name + ' sub-solar longitude wrt node (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.gmtest(bp.ring_sub_solar_longitude(name, reference='obs'),
                   name + ' sub-solar longitude wrt observer (deg)',
                   method='mod360', limit=0.01, radius=1)
        bpt.compare(bp.ring_sub_solar_longitude(name, reference='sun'),
                    0.,
                    name + ' sub-solar longitude wrt Sun (deg)',
                    method='mod360')

        # Incidence, solar elevation
        incidence = bp.ring_center_incidence_angle(name, 'sunward')
        bpt.gmtest(incidence,
                   name + ' center incidence angle, sunward (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.compare(incidence - Scalar.HALFPI, 0.,
                    name + ' center incidence minus 90, sunward (deg)',
                    operator='<', method='degrees')
        bpt.gmtest(bp.ring_center_incidence_angle(name, 'north'),
                   name + ' center incidence angle, north (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.gmtest(bp.ring_center_incidence_angle(name, 'observed'),
                   name + ' center incidence angle, observed (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.gmtest(bp.ring_center_incidence_angle(name, 'prograde'),
                   name + ' center incidence angle, prograde (deg)',
                   limit=0.01, method='degrees', radius=1)

        sunward = bp.ring_incidence_angle(name, 'sunward')
        elevation = bp.ring_elevation(name, direction='sun', pole='sunward')
        generic = bp.incidence_angle(name)

        bpt.gmtest(sunward,
                   name + ' incidence angle, sunward (deg)',
                   limit=0.01, radius=1, method='degrees')
        bpt.compare(sunward - Scalar.HALFPI, 0.,
                    name + ' incidence angle minus 90, sunward (deg)',
                    operator='<=', method='degrees')
        bpt.compare(sunward + elevation, Scalar.HALFPI,
                    name + ' incidence plus solar elevation (deg)',
                    limit=1.e-13, method='degrees')
        bpt.compare(sunward - generic, 0.,
                    name + ' incidence angle, sunward minus generic (deg)',
                    limit=1.e-13, method='degrees')

        northward = bp.ring_incidence_angle(name, 'north')
        bpt.gmtest(northward,
                   name + ' incidence angle, north (deg)',
                   limit=0.01, radius=1, method='degrees')

        prograde = bp.ring_incidence_angle(name, 'prograde')
        if planet in ('JUPITER', 'SATURN', 'NEPTUNE'):
            bpt.compare(northward - prograde, 0.,
                        name + ' incidence angle, north minus prograde (deg)',
                        method='degrees')
        elif planet == 'URANUS':
            bpt.compare(northward + prograde, Scalar.PI,
                        name + ' incidence angle, north plus prograde (deg)',
                        limit=1.e-13, method='degrees')

        incidence0 = bp.ring_incidence_angle(name)
        incidence1 = bp.ring_center_incidence_angle(name)
        bpt.compare(incidence0 - incidence1, 0.,
                    name + ' incidence angle, ring minus center (deg)',
                    limit=0.1, method='degrees')

        # Emission, observer elevation
        bpt.gmtest(bp.ring_center_emission_angle(name, 'sunward'),
                   name + ' center emission angle, sunward (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.gmtest(bp.ring_center_emission_angle(name, 'north'),
                   name + ' center emission angle, north (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.gmtest(bp.ring_center_emission_angle(name, 'prograde'),
                   name + ' center emission angle, prograde (deg)',
                   limit=0.01, method='degrees', radius=1)

        emission = bp.ring_center_emission_angle(name, 'observed')
        bpt.gmtest(emission,
                   name + ' center emission angle, observed (deg)',
                   limit=0.01, method='degrees', radius=1)
        bpt.compare(emission - Scalar.HALFPI, 0.,
                    name + ' center emission minus 90, observed (deg)',
                    operator='<', method='degrees')

        emission = bp.ring_emission_angle(name, 'observed')
        elevation = bp.ring_elevation(name, direction='obs', pole='observed')
        generic = bp.emission_angle(name)

        bpt.gmtest(emission,
                   name + ' emission angle, observed (deg)',
                   limit=0.01, radius=1, method='degrees')
        bpt.compare(emission - Scalar.HALFPI, 0.,
                    name + ' emission angle minus 90, observed (deg)',
                    operator='<', method='degrees')
        bpt.compare(emission + elevation, Scalar.HALFPI,
                    name + ' emission plus observer elevation (deg)',
                    limit=1.e-13, method='degrees')

        sunward = bp.ring_emission_angle(name, 'sunward')
        bpt.compare(sunward - generic, 0.,
                    name + ' emission angle, sunward minus generic (deg)',
                    limit=1.e-13, method='degrees')

        northward = bp.ring_emission_angle(name, 'north')
        bpt.gmtest(northward,
                   name + ' emission angle, north (deg)',
                   limit=0.01, radius=1, method='degrees')

        prograde = bp.ring_emission_angle(name, 'prograde')
        if planet in ('JUPITER', 'SATURN', 'NEPTUNE'):
            bpt.compare(northward - prograde, 0.,
                        name + ' emission angle, north minus prograde (deg)',
                        limit=1.e-13, method='degrees')
        elif planet == 'URANUS':
            bpt.compare(northward + prograde, Scalar.PI,
                        name + ' emission angle, north plus prograde (deg)',
                        limit=1.e-13, method='degrees')

        emission0 = bp.ring_emission_angle(name)
        emission1 = bp.ring_center_emission_angle(name)
        bpt.compare(emission0 - emission1, 0.,
                    name + ' emission angle, ring minus center (deg)',
                    limit=5., method='degrees')

    # Mode tests, Saturn only
    for (planet, name) in bpt.planet_ring_pairs:
        if planet != 'SATURN':
            continue

        test0 = bp.ring_radius(name, 70.e3, 100.e3)
        bpt.gmtest(test0,
                   name + ' radius, modeless, 70-100 km',
                   limit=0.1, radius=1)

        test1 = bp.radial_mode(test0.key, 40, 0., 1000., 0., 0., 100.e3)
        bpt.gmtest(test1, name + ' radius, mode 1, 70-100 kkm',
                   limit=0.1, radius=1)

        test2 = bp.radial_mode(test1.key, 40, 0., -1000., 0., 0., 100.e3)
        bpt.gmtest(test2, name + ' radius, mode 1 canceled, 70-100 kkm',
                   limit=0.1, radius=1)

        bpt.compare(test0, test2,
                    name + ' radius, modeless vs. mode 1 canceled (km)',
                    limit=0.1, radius=1)

        test3 = bp.radial_mode(test1.key, 25, 0., 500., 0., 0., 100.e3)
        bpt.gmtest(test3,
                   name + ' radius, modes 1 and 2, 70-100 kkm',
                   limit=0.1, radius=1)
        bpt.gmtest(bp.ring_longitude(test3.key, 'node'),
                   name + ' longitude, modes 1 and 2, 70-100 kkm (deg)',
                   limit=0.01, method='mod360', radius=1)

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_uv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.ring_names:

        # Get approximate ring spatial scale in km/pixel and deg/pixel
        km_per_los_radian = bp.distance(name) / bp.mu(name)
        if np.all(km_per_los_radian.mask):
            continue

        km_per_pixel = km_per_los_radian.max() * pixel_uv
        (ulimit_km, vlimit_km) = km_per_pixel * 0.001

        deg_per_los_radian = km_per_los_radian / bp.ring_radius(name) * DPR
        deg_per_pixel = deg_per_los_radian.max() * pixel_uv
        (ulimit_deg, vlimit_deg) = deg_per_pixel * 0.001

        # ring_radius
        rad = bp.ring_radius(name)
        drad_duv = rad.d_dlos.chain(bp.dlos_duv)
        (drad_du, drad_dv) = drad_duv.extract_denoms()

        drad = bp_u1.ring_radius(name) - bp_u0.ring_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_du,
                    name + ' radius d/du self-check (km/pix)',
                    limit=ulimit_km, radius=1)

        drad = bp_v1.ring_radius(name) - bp_v0.ring_radius(name)
        bpt.compare(drad.wod/bpt.duv, drad_dv,
                    name + ' radius d/dv self-check (km/pix)',
                    limit=vlimit_km, radius=1)

        # ring_longitude
        lon = bp.ring_longitude(name)
        dlon_duv = lon.d_dlos.chain(bp.dlos_duv)
        (dlon_du, dlon_dv) = dlon_duv.extract_denoms()

        dlon = (bp_u1.ring_longitude(name) - bp_u0.ring_longitude(name)).abs()
        dlon = Scalar.PI - (dlon - Scalar.PI).abs()
        bpt.compare((dlon.wod/bpt.duv - dlon_du).abs().median(), 0.,
                    name + ' longitude d/du self-check (deg/pix)',
                    limit=ulimit_deg, method='degrees')

        dlon = (bp_v1.ring_longitude(name) - bp_v0.ring_longitude(name)).abs()
        dlon = Scalar.PI - (dlon.wod - Scalar.PI).abs()
        bpt.compare((dlon.wod/bpt.duv - dlon_dv).abs().median(), 0.,
                    name + ' longitude d/dv self-check (deg/pix)',
                    limit=vlimit_deg, method='degrees')

        # ring_azimuth
        az = bp.ring_azimuth(name)
        daz_duv = az.d_dlos.chain(bp.dlos_duv)
        (daz_du, daz_dv) = daz_duv.extract_denoms()

        daz = (bp_u1.ring_azimuth(name) - bp_u0.ring_azimuth(name)).abs()
        daz = Scalar.PI - (daz - Scalar.PI).abs()
        bpt.compare((daz.wod/bpt.duv - daz_du).abs().median(), 0.,
                    name + ' azimuth d/du self-check (deg/pix)',
                    limit=ulimit_deg, method='degrees')

        daz = (bp_v1.ring_azimuth(name) - bp_v0.ring_azimuth(name)).abs()
        daz = Scalar.PI - (daz - Scalar.PI).abs()
        bpt.compare((daz.wod/bpt.duv - daz_dv).abs().median(), 0.,
                    name + ' azimuth d/dv self-check (deg/pix)',
                    limit=vlimit_deg, method='degrees')

        # ring_elevation is tested by incidence and emission

register_test_suite('ring', ring_test_suite)

################################################################################
