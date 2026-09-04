##########################################################################################
# programs/gold_master/spheroid.py
##########################################################################################
"""Gold master tests of the surface coordinate backplanes of a spheroidal or ellipsoidal
body: longitude, latitude, and the sub-observer and sub-solar points.
"""

import numpy as np

from polymath         import Scalar
from oops.body        import Body
from oops.constants   import DPR
from programs.gold_master import register_test_suite

def spheroid_test_suite(bpt):
    """Test the surface coordinate backplanes of every body and limb.

    Longitudes are compared to their gold masters relative to the IAU prime meridian, the
    observer, the observer hour angle, the Sun, and the solar hour angle, and in the
    eastward direction; latitudes are compared in both their planetocentric and
    planetographic forms. Longitude limits are scaled by 1/cos(lat), because meridians
    converge toward the poles, and limb tests are given a larger shift radius to suppress
    false positives.

    For each body, the sub-observer and sub-solar longitudes and latitudes are also
    compared to their gold masters. The sub-observer longitude relative to the observer
    and the sub-solar longitude relative to the Sun are each confirmed to be zero, because
    those references are defined by those points. A longitude backplane of a body outside
    the field of view is evaluated once, to confirm that a fully masked backplane behaves
    correctly.

    If bpt.derivs is True, the analytic d/du and d/dv derivatives of the longitude and
    latitude are also compared against numerical derivatives formed from the four
    backplanes offset in u and v by half of bpt.duv. Those comparisons use the median
    absolute difference, and longitude differences are wrapped into the range -180 to 180
    degrees before comparison.

    Parameters:
        bpt (BackplaneTest): The gold master test object, which provides the Backplane to
            evaluate, the lists of surface names to test, and the gmtest() and compare()
            methods that log each result.
    """

    bp = bpt.backplane
    for name in bpt.body_names + bpt.limb_names:

        radius = 1.5 if name in bpt.limb_names else 1.
            # The extra flexibility in the testing of limb calculations seems to
            # reduce the number of false positives.

        # Longitude
        cos_lat = bp.latitude(name).cos().min(builtins=True)
        bpt.gmtest(bp.longitude(name, 'iau'),
                   name + ' longitude, IAU (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, 'obs'),
                   name + ' longitude wrt observer (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, reference='obs', minimum=-180),
                   name + ' longitude wrt observer, minimum -180 (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, 'oha'),
                   name + ' longitude wrt OHA (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, 'sun'),
                   name + ' longitude wrt Sun (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, 'sha'),
                   name + ' longitude wrt SHA (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)
        bpt.gmtest(bp.longitude(name, direction='east'),
                   name + ' longitude eastward (deg)',
                   limit=0.001/cos_lat, method='mod360', radius=radius)

        # Latitude
        bpt.gmtest(bp.latitude(name, lat_type='centric'),
                   name + ' latitude, planetocentric (deg)',
                   limit=0.001, method='degrees', radius=radius)
        bpt.gmtest(bp.latitude(name, lat_type='graphic'),
                   name + ' latitude, planetographic (deg)',
                   limit=0.001, method='degrees', radius=radius)

    for name in bpt.body_names:

        # Sub-observer longitude and latitude
        cos_lat = bp.sub_observer_latitude(name).cos().mean(builtins=True)
        bpt.gmtest(bp.sub_observer_longitude(name, reference='iau'),
                   name + ' sub-observer longitude, IAU (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.gmtest(bp.sub_observer_longitude(name, reference='sun', minimum=-180),
                   name + ' sub-observer longitude wrt Sun (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.compare(bp.sub_observer_longitude(name, reference='obs', minimum=-180),
                    0.,
                    name + ' sub-observer longitude wrt observer (deg)',
                    method='mod360')

        bpt.gmtest(bp.sub_observer_latitude(name, lat_type='centric'),
                   name + ' sub-observer latitude, planetocentric (deg)',
                   limit=0.001, method='degrees')
        bpt.gmtest(bp.sub_observer_latitude(name, lat_type='graphic'),
                   name + ' sub-observer latitude, planetographic (deg)',
                   limit=0.001, method='degrees')

        # Sub-solar longitude and latitude
        cos_lat = bp.sub_solar_latitude(name).cos().mean(builtins=True)
        bpt.gmtest(bp.sub_solar_longitude(name, reference='iau'),
                   name + ' sub-solar longitude wrt IAU (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.gmtest(bp.sub_solar_longitude(name, reference='obs', minimum=-180),
                   name + ' sub-solar longitude wrt observer (deg)',
                   limit=0.001/cos_lat, method='mod360')
        bpt.compare(bp.sub_solar_longitude(name, reference='sun', minimum=-180),
                    0.,
                    name + ' sub-solar longitude wrt Sun (deg)',
                    method='mod360')

        bpt.gmtest(bp.sub_solar_latitude(name, lat_type='centric'),
                   name + ' sub-solar latitude, planetocentric (deg)',
                   limit=0.001, method='degrees')
        bpt.gmtest(bp.sub_solar_latitude(name, lat_type='graphic'),
                   name + ' sub-solar latitude, planetographic (deg)',
                   limit=0.001, method='degrees')

    # Test of an empty backplane
    for (planet, name) in bpt.planet_moon_pairs:
        if planet != 'PLUTO':
            bpt.compare(bp.longitude('STYX'),
                        0.,
                        'Styx longitude (deg)')
            break   # no need to repeat this test!

    # Derivative tests
    if bpt.derivs:
      (bp, bp_u0, bp_u1, bp_v0, bp_v1) = bpt.backplanes
      pixel_duv = np.abs(bp.obs.fov.uv_scale.vals)

      for name in bpt.body_names:

        # Get approximate projected surface scale in degrees lat/lon per pixel
        km_per_fov_radian = bp.distance(name) / bp.mu(name)
        rad_per_fov_radian = km_per_fov_radian / Body.lookup(name).radius
        deg_per_fov_radian = rad_per_fov_radian * DPR

        # longitude
        cos_lat = bp.latitude(name).cos()
        (ulimit,
         vlimit) = (deg_per_fov_radian/cos_lat).median() * pixel_duv * 0.01

        lon = bp.longitude(name)
        dlon_duv = lon.d_dlos.chain(bp.dlos_duv)
        (dlon_du, dlon_dv) = dlon_duv.extract_denoms()

        dlon = (bp_u1.longitude(name) - bp_u0.longitude(name)).abs()
        dlon = Scalar.PI - (dlon.wod - Scalar.PI).abs()
        if not np.all(dlon.mask):
            bpt.compare((dlon/bpt.duv - dlon_du).abs().median(), 0.,
                        name + ' longitude d/du self-check (deg/pix)',
                        limit=ulimit, method='degrees')

        dlon = (bp_v1.longitude(name) - bp_v0.longitude(name)).abs()
        dlon = Scalar.PI - (dlon.wod - Scalar.PI).abs()
        if not np.all(dlon.mask):
            bpt.compare((dlon/bpt.duv - dlon_dv).abs().median(), 0.,
                        name + ' longitude d/dv self-check (deg/pix)',
                        limit=vlimit, method='degrees')

        (ulimit, vlimit) = deg_per_fov_radian.median() * pixel_duv * 0.01

        # latitude
        lat = bp.latitude(name)
        dlat_duv = lat.d_dlos.chain(bp.dlos_duv)
        (dlat_du, dlat_dv) = dlat_duv.extract_denoms()

        dlat = bp_u1.latitude(name) - bp_u0.latitude(name)
        if not np.all(dlat.mask):
            bpt.compare((dlat.wod/bpt.duv - dlat_du).abs().median(), 0.,
                        name + ' latitude d/du self-check (deg/pix)',
                        limit=ulimit, radius=1, method='degrees')

        dlat = bp_v1.latitude(name) - bp_v0.latitude(name)
        if not np.all(dlat.mask):
            bpt.compare((dlat.wod/bpt.duv - dlat_dv).abs().median(), 0.,
                        name + ' latitude d/dv self-check (deg/pix)',
                        limit=vlimit, radius=1, method='degrees')

register_test_suite('spheroid', spheroid_test_suite)

##########################################################################################
