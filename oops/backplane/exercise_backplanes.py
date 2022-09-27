################################################################################
# oops/backplane/exercise_backplanes.py
################################################################################

from __future__ import print_function

import numpy as np
#import os.path

#from polymath import Vector3

from oops.backplane    import Backplane
#from oops.body         import Body
from oops.event        import Event
from oops.meshgrid     import Meshgrid
from oops.surface.ansa import Ansa
from oops.constants    import HALFPI, DPR
import oops.config as config

#===============================================================================
def exercise_backplanes(obs, printing, logging, saving, undersample=16,
                                  use_inventory=False, inventory_border=2):
    """Generates info from every backplane."""

    import numbers
    from PIL import Image

    #===========================================================================
    def save_image(image, filename, lo=None, hi=None):
        """Save an image file of a 2-D array.

        Input:
            image       a 2-D array.
            filename    the name of the output file, which should end with the
                        type, e.g., '.png' or '.jpg'
            lo          the array value to map to black; if None, then the
                        minimum value in the array is used.
            hi          the array value to map to white; if None, then the
                        maximum value in the array is used.
        """

        image = np.asfarray(image)

        if lo is None:
            lo = image.min()

        if hi is None:
            hi = image.max()

        if hi == lo:
            bytes = np.zeros(image.shape, dtype='uint8')
        else:
            scaled = (image[::-1] - lo) / float(hi - lo)
            bytes = (256.*scaled).clip(0,255).astype('uint8')

        im = Image.frombytes('L', (bytes.shape[1], bytes.shape[0]), bytes)
        im.save(filename)

    #===========================================================================
    def show_info(title, array):
        """Internal method to print summary information and display images as
        desired.
        """

        if not printing and not saving:
            return

        if printing:
            print(title)

        # Scalar summary
        if isinstance(array, numbers.Number):
            print('  ', array)

        # Mask summary
        elif type(array.vals) == bool or \
                (isinstance(array.vals, np.ndarray) and \
                 array.vals.dtype == np.dtype('bool')):
            count = np.sum(array.vals)
            total = np.size(array.vals)
            percent = int(count / float(total) * 100. + 0.5)
            print('  ', (count, total-count),
                        (percent, 100-percent), '(True, False pixels)')
            minval = 0.
            maxval = 1.

        # Unmasked backplane summary
        elif array.mask is False:
            minval = np.min(array.vals)
            maxval = np.max(array.vals)
            if minval == maxval:
                print('  ', minval)
            else:
                print('  ', (minval, maxval), '(min, max)')

        # Masked backplane summary
        else:
            print('  ', (array.min().as_builtin(),
                         array.max().as_builtin()), '(masked min, max)')
            total = np.size(array.mask)
            masked = np.sum(array.mask)
            percent = int(masked / float(total) * 100. + 0.5)
            print('  ', (masked, total-masked),
                        (percent, 100-percent), '(masked, unmasked pixels)')

            if total == masked:
                minval = np.min(array.vals)
                maxval = np.max(array.vals)
            else:
                minval = array.min().as_builtin()
                maxval = array.max().as_builtin()

        if saving and array.shape != ():
            if minval == maxval:
                maxval += 1.

            image = array.vals.copy()
            image[array.mask] = minval - 0.05 * (maxval - minval)

            filename = 'backplane-' + title + '.png'
            filename = filename.replace(':','_')
            filename = filename.replace('/','_')
            filename = filename.replace(' ','_')
            filename = filename.replace('(','_')
            filename = filename.replace(')','_')
            filename = filename.replace('[','_')
            filename = filename.replace(']','_')
            filename = filename.replace('&','_')
            filename = filename.replace(',','_')
            filename = filename.replace('-','_')
            filename = filename.replace('__','_')
            filename = filename.replace('__','_')
            filename = filename.replace('_.','.')
            save_image(image, filename)

    if printing and logging: config.LOGGING.on('        ')

    if printing:
        print()

    meshgrid = Meshgrid.for_fov(obs.fov, undersample=undersample, swap=True)

    if use_inventory:
        bp = Backplane(obs, meshgrid, inventory={})
    else:
        bp = Backplane(obs, meshgrid, inventory=None)

    ########################

    if printing: print('\n********* right ascension')

    test = bp.right_ascension(apparent=False)
    show_info('Right ascension (deg, actual)', test * DPR)

    test = bp.right_ascension(apparent=True)
    show_info('Right ascension (deg, apparent)', test * DPR)

    test = bp.center_right_ascension('saturn', apparent=False)
    show_info('Right ascension of Saturn (deg, actual)', test * DPR)

    test = bp.center_right_ascension('saturn', apparent=True)
    show_info('Right ascension of Saturn (deg, apparent)', test * DPR)

    test = bp.center_right_ascension('epimetheus', apparent=False)
    show_info('Right ascension of Epimetheus (deg, actual)',
                                                    test * DPR)

    test = bp.center_right_ascension('epimetheus', apparent=True)
    show_info('Right ascension of Epimetheus (deg, apparent)',
                                                    test * DPR)

    ########################

    if printing: print('\n********* declination')

    test = bp.declination(apparent=False)
    show_info('Declination (deg, actual)', test * DPR)

    test = bp.declination(apparent=True)
    show_info('Declination (deg, apparent)', test * DPR)

    test = bp.center_declination('saturn', apparent=False)
    show_info('Declination of Saturn (deg, actual)', test * DPR)

    test = bp.center_declination('saturn', apparent=True)
    show_info('Declination of Saturn (deg, apparent)', test * DPR)

    test = bp.center_declination('epimetheus', apparent=False)
    show_info('Declination of Epimetheus (deg, actual)',
                                                    test * DPR)

    test = bp.center_declination('epimetheus', apparent=True)
    show_info('Declination of Epimetheus (deg, apparent)',
                                                    test * DPR)

    ########################

    if printing: print('\n********* celestial and polar angles')

    test = bp.celestial_north_angle()
    show_info('Celestial north angle (deg)', test * DPR)

    test = bp.celestial_east_angle()
    show_info('Celestial east angle (deg)', test * DPR)

    test = bp.pole_clock_angle('saturn')
    show_info('Saturn pole clock angle (deg)', test * DPR)

    test = bp.pole_position_angle('saturn')
    show_info('Saturn pole position angle (deg)', test * DPR)

    ########################

    if printing: print('\n********* observer distances')

    test = bp.distance('saturn')
    show_info('Distance observer to Saturn (km)', test)

    test = bp.distance('saturn', direction='dep')
    show_info('Distance observer to Saturn via dep (km)', test)

    test = bp.center_distance('saturn')
    show_info('Distance observer to Saturn center (km)', test)

    test = bp.distance('saturn_main_rings')
    show_info('Distance observer to rings (km)', test)

    test = bp.center_distance('saturn_main_rings')
    show_info('Distance observer to ring center (km)', test)

    test = bp.distance('saturn:limb')
    show_info('Distance observer to Saturn limb (km)', test)

    test = bp.distance('saturn:ansa')
    show_info('Distance observer to ansa (km)', test)

    test = bp.distance('epimetheus')
    show_info('Distance observer to Epimetheus (km)', test)

    test = bp.center_distance('epimetheus')
    show_info('Distance observer to Epimetheus center (km)', test)

    ########################

    if printing: print('\n********* Sun distances')

    test = bp.distance('saturn', direction='arr')
    show_info('Distance Sun to Saturn, arrival (km)', test)

    test = bp.distance(('sun', 'saturn'), direction='dep')
    show_info('Distance Sun to Saturn, departure (km)', test)

    test = bp.center_distance('saturn', direction='arr')
    show_info('Distance Sun to Saturn center, arrival (km)', test)

    test = bp.center_distance(('sun', 'saturn'), direction='dep')
    show_info('Distance Sun to Saturn center, departure (km)', test)

    test = bp.distance('saturn_main_rings', direction='arr')
    show_info('Distance Sun to rings, arrival (km)', test)

    test = bp.distance(('sun', 'saturn_main_rings'), direction='dep')
    show_info('Distance Sun to rings, departure (km)', test)

    test = bp.center_distance('saturn_main_rings', direction='arr')
    show_info('Distance Sun to ring center, arrival (km)', test)

    test = bp.center_distance(('sun', 'saturn_main_rings'), direction='dep')
    show_info('Distance Sun to ring center, departure (km)', test)

    test = bp.distance('saturn:ansa', direction='arr')
    show_info('Distance Sun to ansa (km)', test)

    test = bp.distance('saturn:limb', direction='arr')
    show_info('Distance Sun to limb (km)', test)

    ########################

    if printing: print('\n********* observer light time')

    test = bp.light_time('saturn')
    show_info('Light-time observer to Saturn (sec)', test)

    test = bp.light_time('saturn', direction='dep')
    show_info('Light-time observer to Saturn via dep (sec)', test)

    test = bp.light_time('saturn_main_rings')
    show_info('Light-time observer to rings (sec)', test)

    test = bp.light_time('saturn:limb')
    show_info('Light-time observer to limb (sec)', test)

    test = bp.light_time('saturn:ansa')
    show_info('Light-time observer to ansa (sec)', test)

    test = bp.center_light_time('saturn')
    show_info('Light-time observer to Saturn center (sec)', test)

    test = bp.center_light_time('saturn_main_rings')
    show_info('Light-time observer to ring center (sec)', test)

    test = bp.light_time('epimetheus')
    show_info('Light-time observer to Epimetheus (sec)', test)

    test = bp.center_light_time('epimetheus')
    show_info('Light-time observer to Epimetheus center (sec)', test)

    ########################

    if printing: print('\n********* Sun light time')

    test = bp.light_time('saturn', direction='arr')
    show_info('Light-time Sun to Saturn via arr (sec)', test)

    test = bp.light_time(('sun', 'saturn'))
    show_info('Light-time Sun to Saturn via sun-saturn (sec)', test)

    test = bp.center_light_time(('sun', 'saturn'))
    show_info('Light-time Sun to Saturn at centers (sec)', test)

    test = bp.light_time('saturn_main_rings', direction='arr')
    show_info('Light-time Sun to rings (sec)', test)

    test = bp.center_light_time(('sun', 'saturn_main_rings'))
    show_info('Light-time Sun to rings at centers (sec)', test)

    test = bp.light_time('saturn:ansa', direction='arr')
    show_info('Light-time Sun to ansa (sec)', test)

    test = bp.light_time('saturn:limb', direction='arr')
    show_info('Light-time Sun to limb (sec)', test)

    ########################

    if printing: print('\n********* event time')

    test = bp.event_time(())
    show_info('Event time at Cassini (sec, TDB)', test)

    test = bp.event_time('saturn')
    show_info('Event time at Saturn (sec, TDB)', test)

    test = bp.event_time('saturn_main_rings')
    show_info('Event time at rings (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus (sec, TDB)', test)

    test = bp.center_time(())
    show_info('Event time at Cassini center (sec, TDB)', test)

    test = bp.center_time('saturn')
    show_info('Event time at Saturn (sec, TDB)', test)

    test = bp.center_time('saturn_main_rings')
    show_info(' Event time at ring center (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus (sec, TDB)', test)

    test = bp.event_time('epimetheus')
    show_info('Event time at Epimetheus center (sec, TDB)', test)

    ########################

    if printing: print('\n********* resolution')

    test = bp.resolution('saturn', 'u')
    show_info('Saturn resolution along u axis (km)', test)

    test = bp.resolution('saturn', 'v')
    show_info('Saturn resolution along v axis (km)', test)

    test = bp.center_resolution('saturn', 'u')
    show_info('Saturn center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn', 'v')
    show_info('Saturn center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn')
    show_info('Saturn finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn')
    show_info('Saturn coarsest resolution (km)', test)

    test = bp.resolution('epimetheus', 'u')
    show_info('Epimetheus resolution along u axis (km)', test)

    test = bp.resolution('epimetheus', 'v')
    show_info('Epimetheus resolution along v axis (km)', test)

    test = bp.center_resolution('epimetheus', 'u')
    show_info('Epimetheus center resolution along u axis (km)', test)

    test = bp.center_resolution('epimetheus', 'v')
    show_info('Epimetheus center resolution along v axis (km)', test)

    test = bp.finest_resolution('epimetheus')
    show_info('Epimetheus finest resolution (km)', test)

    test = bp.coarsest_resolution('epimetheus')
    show_info('Epimetheus coarsest resolution (km)', test)

    test = bp.resolution('saturn_main_rings', 'u')
    show_info('Ring resolution along u axis (km)', test)

    test = bp.resolution('saturn_main_rings', 'v')
    show_info('Ring resolution along v axis (km)', test)

    test = bp.center_resolution('saturn_main_rings', 'u')
    show_info('Ring center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn_main_rings', 'v')
    show_info('Ring center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn_main_rings')
    show_info('Ring finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn_main_rings')
    show_info('Ring coarsest resolution (km)', test)

    test = bp.ring_radial_resolution('saturn_main_rings')
    show_info('Ring radial resolution (km)', test)

    test = bp.ring_angular_resolution('saturn_main_rings')
    show_info('Ring angular resolution (deg)', test * DPR)

    radii = bp.ring_radius('saturn_main_rings')
    show_info('Ring angular resolution (km)', test * radii)

    test = bp.resolution('saturn:ansa', 'u')
    show_info('Ansa resolution along u axis (km)', test)

    test = bp.resolution('saturn:ansa', 'v')
    show_info('Ansa resolution along v axis (km)', test)

    test = bp.center_resolution('saturn:ansa', 'u')
    show_info('Ansa center resolution along u axis (km)', test)

    test = bp.center_resolution('saturn:ansa', 'v')
    show_info('Ansa center resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn:ansa')
    show_info('Ansa finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn:ansa')
    show_info('Ansa coarsest resolution (km)', test)

    test = bp.ansa_radial_resolution('saturn:ansa')
    show_info('Ansa radial resolution (km)', test)

    test = bp.ansa_vertical_resolution('saturn:ansa')
    show_info('Ansa vertical resolution (km)', test)

    test = bp.resolution('saturn:limb', 'u')
    show_info('Limb resolution along u axis (km)', test)

    test = bp.resolution('saturn:limb', 'v')
    show_info('Limb resolution along v axis (km)', test)

    test = bp.resolution('saturn:limb', 'u')
    show_info('Limb resolution along u axis (km)', test)

    test = bp.resolution('saturn:limb', 'v')
    show_info('Limb resolution along v axis (km)', test)

    test = bp.finest_resolution('saturn:limb')
    show_info('Limb finest resolution (km)', test)

    test = bp.coarsest_resolution('saturn:limb')
    show_info('Limb coarsest resolution (km)', test)

    ########################

    if printing: print('\n********* surface latitude')

    test = bp.latitude('saturn', lat_type='centric')
    show_info('Saturn latitude, planetocentric (deg)', test * DPR)

    test = bp.latitude('saturn', lat_type='squashed')
    show_info('Saturn latitude, squashed (deg)', test * DPR)

    test = bp.latitude('saturn', lat_type='graphic')
    show_info('Saturn latitude, planetographic (deg)', test * DPR)

    test = bp.sub_observer_latitude('saturn')
    show_info('Saturn sub-observer latitude (deg)', test * DPR)

    test = bp.sub_solar_latitude('saturn')
    show_info('Saturn sub-solar latitude (deg)', test * DPR)

    test = bp.latitude('epimetheus', lat_type='centric')
    show_info('Epimetheus latitude, planetocentric (deg)', test * DPR)

    test = bp.latitude('epimetheus', lat_type='squashed')
    show_info('Epimetheus latitude, squashed (deg)', test * DPR)

    test = bp.latitude('epimetheus', lat_type='graphic')
    show_info('Epimetheus latitude, planetographic (deg)', test * DPR)

    test = bp.sub_observer_latitude('epimetheus')
    show_info('Epimetheus sub-observer latitude (deg)', test * DPR)

    test = bp.sub_solar_latitude('epimetheus')
    show_info('Epimetheus sub-solar latitude (deg)', test * DPR)

    ########################

    if printing: print('\n********* surface longitude')

    test = bp.longitude('saturn')
    show_info('Saturn longitude (deg)', test * DPR)

    test = bp.longitude('saturn', reference='iau')
    show_info('Saturn longitude wrt IAU frame (deg)',
                                                test * DPR)

    test = bp.longitude('saturn', lon_type='centric')
    show_info('Saturn longitude centric (deg)', test * DPR)

    test = bp.longitude('saturn', lon_type='graphic')
    show_info('Saturn longitude graphic (deg)', test * DPR)

    test = bp.longitude('saturn', lon_type='squashed')
    show_info('Saturn longitude squashed (deg)', test * DPR)

    test = bp.longitude('saturn', direction='east')
    show_info('Saturn longitude eastward (deg)', test * DPR)

    test = bp.longitude('saturn', minimum=-180)
    show_info('Saturn longitude with -180 minimum (deg)',
                                                test * DPR)

    test = bp.longitude('saturn', reference='iau', minimum=-180)
    show_info('Saturn longitude wrt IAU frame with -180 minimum (deg)',
                                                test * DPR)

    test = bp.longitude('saturn', reference='sun')
    show_info('Saturn longitude wrt Sun (deg)', test * DPR)

    test = bp.longitude('saturn', reference='sha')
    show_info('Saturn longitude wrt SHA (deg)', test * DPR)

    test = bp.longitude('saturn', reference='obs')
    show_info('Saturn longitude wrt observer (deg)',
                                                test * DPR)

    test = bp.longitude('saturn', reference='oha')
    show_info('Saturn longitude wrt OHA (deg)', test * DPR)

    test = bp.sub_observer_longitude('saturn', reference='iau')
    show_info('Saturn sub-observer longitude wrt IAU (deg)',
                                                test * DPR)

    test = bp.sub_observer_longitude('saturn', reference='sun', minimum=-180)
    show_info('Saturn sub-observer longitude wrt Sun (deg)',
                                                test * DPR)

    test = bp.sub_observer_longitude('saturn', reference='obs', minimum=-180)
    show_info('Saturn sub-observer longitude wrt observer (deg)',
                                                test * DPR)

    test = bp.sub_solar_longitude('saturn', reference='iau')
    show_info('Saturn sub-solar longitude wrt IAU (deg)',
                                                test * DPR)

    test = bp.sub_solar_longitude('saturn', reference='obs', minimum=-180)
    show_info('Saturn sub-solar longitude wrt observer (deg)',
                                                test * DPR)

    test = bp.sub_solar_longitude('saturn', reference='sun', minimum=-180)
    show_info('Saturn sub-solar longitude wrt Sun (deg)',
                                                test * DPR)

    test = bp.longitude('epimetheus')
    show_info('Epimetheus longitude (deg)', test * DPR)

    test = bp.sub_observer_longitude('epimetheus')
    show_info('Epimetheus sub-observer longitude (deg)', test * DPR)

    test = bp.sub_solar_longitude('epimetheus')
    show_info('Epimetheus sub-solar longitude (deg)', test * DPR)

# Used for testing other images
#     test = bp.longitude('enceladus')
#     show_info('Enceladus longitude (deg)', test * DPR)
#
#     test = bp.sub_observer_longitude('enceladus')
#     show_info('Enceladus sub-observer longitude (deg)', test * DPR)
#
#     test = bp.sub_solar_longitude('enceladus')
#     show_info('Enceladus sub-solar longitude (deg)', test * DPR)

    ########################

    if printing: print('\n********* surface incidence, emission, phase')

    test = bp.phase_angle('saturn')
    show_info('Saturn phase angle (deg)', test * DPR)

    test = bp.scattering_angle('saturn')
    show_info('Saturn scattering angle (deg)', test * DPR)

    test = bp.incidence_angle('saturn')
    show_info('Saturn incidence angle (deg)', test * DPR)

    test = bp.emission_angle('saturn')
    show_info('Saturn emission angle (deg)', test * DPR)

    test = bp.lambert_law('saturn')
    show_info('Saturn as a Lambert law', test)

    ########################

    if printing: print('\n********* ring radius, radial modes')

    test = bp.ring_radius('saturn_main_rings')
    show_info('Ring radius (km)', test)

    test0 = bp.ring_radius('saturn_main_rings', 70.e3, 100.e3)
    show_info('Ring radius, 70-100 kkm (km)', test0)

    test1 = bp.radial_mode(test0.key, 40, 0., 1000., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, mode 1 (km)', test1)

    test = bp.radial_mode(test1.key, 40, 0., -1000., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, mode 1 canceled (km)', test)

    test2 = bp.radial_mode(test1.key, 25, 0., 500., 0., 0., 100.e3)
    show_info('Ring radius, 70-100 kkm, modes 1 and 2 (km)', test2)

    test = bp.ring_radius('saturn_main_rings').without_mask()
    show_info('Ring radius unmasked (km)', test)

    ########################

    if printing: print('\n********* ring longitude, azimuth')

    test = bp.ring_longitude('saturn_main_rings', reference='node')
    show_info('Ring longitude wrt node (deg)', test * DPR)

    test = bp.ring_longitude('saturn_main_rings', 'node', 70.e3, 100.e3)
    show_info('Ring longitude wrt node, 70-100 kkm (deg)', test * DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='aries')
    show_info('Ring longitude wrt Aries (deg)', test * DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='obs')
    show_info('Ring longitude wrt observer (deg)',
                                                test * DPR)

    test = bp.ring_azimuth('saturn_main_rings', 'obs')
    show_info('Ring azimuth wrt observer (deg)', test * DPR)

    test = bp.ring_azimuth('saturn:ring', 'obs')
    show_info('Ring azimuth wrt observer (deg)', test * DPR)

    compare = bp.ring_longitude('saturn_main_rings', 'obs')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt observer (deg)',
                                                        diff * DPR)

    test = bp.ring_azimuth('saturn:ring', 'obs')
    show_info('Ring azimuth wrt observer, unmasked (deg)', test * DPR)

    compare = bp.ring_longitude('saturn:ring', 'obs')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt observer, unmasked (deg)',
                                                        diff * DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='oha')
    show_info('Ring longitude wrt OHA (deg)', test * DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='sun')
    show_info('Ring longitude wrt Sun (deg)', test * DPR)

    test = bp.ring_azimuth('saturn_main_rings', reference='sun')
    show_info('Ring azimuth wrt Sun (deg)', test * DPR)

    compare = bp.ring_longitude('saturn_main_rings', 'sun')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt Sun (deg)',
                                                        diff * DPR)

    test = bp.ring_azimuth('saturn:ring', reference='sun')
    show_info('Ring azimuth wrt Sun, unmasked (deg)', test * DPR)

    compare = bp.ring_longitude('saturn:ring', 'sun')
    diff = test - compare
    show_info('Ring azimuth minus longitude wrt Sun, unmasked (deg)',
                                                        diff * DPR)

    test = bp.ring_longitude('saturn_main_rings', reference='sha')
    show_info('Ring longitude wrt SHA (deg)', test * DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'node')
    show_info('Ring sub-observer longitude wrt node (deg)',
                                                        test * DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'aries')
    show_info('Ring sub-observer longitude wrt Aries (deg)',
                                                        test * DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'sun')
    show_info('Ring sub-observer longitude wrt Sun (deg)', test * DPR)

    test = bp.ring_sub_observer_longitude('saturn_main_rings', 'obs')
    show_info('Ring sub-observer longitude wrt observer (deg)',
                                                        test * DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'node')
    show_info('Ring sub-solar longitude wrt node (deg)', test * DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'aries')
    show_info('Ring sub-solar longitude wrt Aries (deg)', test * DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'sun')
    show_info('Ring sub-solar longitude wrt Sun (deg)', test * DPR)

    test = bp.ring_sub_solar_longitude('saturn_main_rings', 'obs')
    show_info('Ring sub-solar longitude wrt observer (deg)',
                                                        test * DPR)

    ########################

    if printing: print('\n********* ring phase angle')

    test = bp.phase_angle('saturn_main_rings')
    show_info('Ring phase angle (deg)', test * DPR)

    test = bp.sub_observer_longitude('saturn_main_rings', 'sun', minimum=-180)
    show_info('Ring observer-sun longitude (deg)', test * DPR)

    ########################

    if printing: print('\n********* ring incidence, solar elevation')

    test = bp.ring_incidence_angle('saturn_main_rings', 'sunward')
    show_info('Ring incidence angle, sunward (deg)', test * DPR)

    test = bp.ring_incidence_angle('saturn_main_rings', 'north')
    show_info('Ring incidence angle, north (deg)', test * DPR)

    test = bp.ring_incidence_angle('saturn_main_rings', 'prograde')
    show_info('Ring incidence angle, prograde (deg)', test * DPR)

    test = bp.incidence_angle('saturn_main_rings')
    show_info('Ring incidence angle via incidence() (deg)',
                                                        test * DPR)

    test = bp.ring_elevation('saturn_main_rings', reference='sun')
    show_info('Ring elevation wrt Sun (deg)', test * DPR)

    compare = bp.ring_incidence_angle('saturn_main_rings', 'north')
    diff = test + compare
    show_info('Ring elevation wrt Sun plus north incidence (deg)',
                                                        diff * DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'sunward')
    show_info('Ring center incidence angle, sunward (deg)',
                                                        test * DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'north')
    show_info('Ring center incidence angle, north (deg)', test * DPR)

    test = bp.ring_center_incidence_angle('saturn_main_rings', 'prograde')
    show_info('Ring center incidence angle, prograde (deg)',
                                                        test * DPR)

    test = bp.ring_elevation('saturn:ring', reference='sun')
    show_info('Ring elevation wrt Sun, unmasked (deg)', test * DPR)

    compare = bp.ring_incidence_angle('saturn:ring', 'north')
    diff = test + compare
    show_info('Ring elevation wrt Sun plus north incidence, unmasked (deg)',
                                                        diff * DPR)

    ########################

    if printing: print('\n********* ring emission, observer elevation')

    test = bp.ring_emission_angle('saturn_main_rings', 'sunward')
    show_info('Ring emission angle, sunward (deg)', test * DPR)

    test = bp.ring_emission_angle('saturn_main_rings', 'north')
    show_info('Ring emission angle, north (deg)', test * DPR)

    test = bp.ring_emission_angle('saturn_main_rings', 'prograde')
    show_info('Ring emission angle, prograde (deg)', test * DPR)

    test = bp.emission_angle('saturn_main_rings')
    show_info('Ring emission angle via emission() (deg)', test * DPR)

    test = bp.ring_elevation('saturn_main_rings', reference='obs')
    show_info('Ring elevation wrt observer (deg)', test * DPR)

    compare = bp.ring_emission_angle('saturn_main_rings', 'north')
    diff = test + compare
    show_info('Ring elevation wrt observer plus north emission (deg)',
                                                        diff * DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'sunward')
    show_info('Ring center emission angle, sunward (deg)', test * DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'north')
    show_info('Ring center emission angle, north (deg)', test * DPR)

    test = bp.ring_center_emission_angle('saturn_main_rings', 'prograde')
    show_info('Ring center emission angle, prograde (deg)',
                                                        test * DPR)

    test = bp.ring_elevation('saturn:ring', reference='obs')
    show_info('Ring elevation wrt observer, unmasked (deg)',
                                                        test * DPR)

    compare = bp.ring_emission_angle('saturn:ring', 'north')
    diff = test + compare
    show_info('Ring elevation wrt observer plus north emission, unmasked (deg)',
                                                        diff * DPR)

    ########################

    if printing: print('\n********* ansa geometry')

    test = bp.ansa_radius('saturn:ansa')
    show_info('Ansa radius (km)', test)

    test = bp.ansa_altitude('saturn:ansa')
    show_info('Ansa altitude (km)', test)

    test = bp.ansa_longitude('saturn:ansa', 'node')
    show_info('Ansa longitude wrt node (deg)', test * DPR)

    test = bp.ansa_longitude('saturn:ansa', 'aries')
    show_info('Ansa longitude wrt Aries (deg)', test * DPR)

    test = bp.ansa_longitude('saturn:ansa', 'obs')
    show_info('Ansa longitude wrt observer (deg)', test * DPR)

    test = bp.ansa_longitude('saturn:ansa', 'oha')
    show_info('Ansa longitude wrt OHA (deg)', test * DPR)

    test = bp.ansa_longitude('saturn:ansa', 'sun')
    show_info('Ansa longitude wrt Sun (deg)', test * DPR)

    test = bp.ansa_longitude('saturn:ansa', 'sha')
    show_info('Ansa longitude wrt SHA (deg)', test * DPR)

    ########################

    if printing: print('\n********* limb altitude')

    test = bp.limb_altitude('saturn:limb')
    show_info('Limb altitude (km)', test)

    ########################

    if printing: print('\n********* limb longitude')

    test = bp.longitude('saturn:limb', 'iau')
    show_info('Limb longitude wrt IAU (deg)', test * DPR)

    test = bp.longitude('saturn:limb', 'obs')
    show_info('Limb longitude wrt observer (deg)', test * DPR)

    test = bp.longitude('saturn:limb', reference='obs', minimum=-180)
    show_info('Limb longitude wrt observer, -180 (deg)',
                                                    test * DPR)

    test = bp.longitude('saturn:limb', 'oha')
    show_info('Limb longitude wrt OHA (deg)', test * DPR)

    test = bp.longitude('saturn:limb', 'sun')
    show_info('Limb longitude wrt Sun (deg)', test * DPR)

    test = bp.longitude('saturn:limb', 'sha')
    show_info('Limb longitude wrt SHA (deg)', test * DPR)

    ########################

    if printing: print('\n********* limb latitude')

    test = bp.latitude('saturn:limb', lat_type='centric')
    show_info('Limb planetocentric latitude (deg)', test * DPR)

    test = bp.latitude('saturn:limb', lat_type='squashed')
    show_info('Limb squashed latitude (deg)', test * DPR)

    test = bp.latitude('saturn:limb', lat_type='graphic')
    show_info('Limb planetographic latitude (deg)', test * DPR)

    ########################

    if printing: print('\n********* orbit longitude')

    test = bp.orbit_longitude('epimetheus', reference='obs')
    show_info('Epimetheus orbit longitude wrt observer (deg)', test * DPR)

    test = bp.orbit_longitude('epimetheus', reference='oha')
    show_info('Epimetheus orbit longitude wrt OHA (deg)', test * DPR)

    test = bp.orbit_longitude('epimetheus', reference='sun')
    show_info('Epimetheus orbit longitude wrt Sun (deg)', test * DPR)

    test = bp.orbit_longitude('epimetheus', reference='sha')
    show_info('Epimetheus orbit longitude wrt SHA (deg)', test * DPR)

    test = bp.orbit_longitude('epimetheus', reference='aries')
    show_info('Epimetheus orbit longitude wrt Aries (deg)', test * DPR)

    test = bp.orbit_longitude('epimetheus', reference='node')
    show_info('Epimetheus orbit longitude wrt node (deg)', test * DPR)

    ########################

    if printing: print('\n********* masks')

    test = bp.where_intercepted('saturn')
    show_info('Mask of Saturn intercepted', test)

    test = bp.evaluate(('where_intercepted', 'saturn'))
    show_info('Mask of Saturn intercepted via evaluate()', test)

    test = bp.where_sunward('saturn')
    show_info('Mask of Saturn sunward', test)

    test = bp.evaluate(('where_sunward', 'saturn'))
    show_info('Mask of Saturn sunward via evaluate()', test)

    test = bp.where_below(('incidence_angle', 'saturn'), HALFPI)
    show_info('Mask of Saturn sunward via where_below()', test)

    test = bp.where_antisunward('saturn')
    show_info('Mask of Saturn anti-sunward', test)

    test = bp.where_above(('incidence_angle', 'saturn'), HALFPI)
    show_info('Mask of Saturn anti-sunward via where_above()', test)

    test = bp.where_between(('incidence_angle', 'saturn'), HALFPI,3.2)
    show_info('Mask of Saturn anti-sunward via where_between()', test)

    test = bp.where_in_front('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn in front of rings', test)

    test = bp.where_in_back('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn behind rings', test)

    test = bp.where_inside_shadow('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn in shadow of rings', test)

    test = bp.where_outside_shadow('saturn', 'saturn_main_rings')
    show_info('Mask of Saturn outside shadow of rings', test)

    test = bp.where_intercepted('saturn_main_rings')
    show_info('Mask of rings intercepted', test)

    test = bp.where_in_front('saturn_main_rings', 'saturn')
    show_info('Mask of rings in front of Saturn', test)

    test = bp.where_in_back('saturn_main_rings', 'saturn')
    show_info('Mask of rings behind Saturn', test)

    test = bp.where_inside_shadow('saturn_main_rings', 'saturn')
    show_info('Mask of rings in shadow of Saturn', test)

    test = bp.where_outside_shadow('saturn_main_rings', 'saturn')
    show_info('Mask of rings outside shadow of Saturn', test)

    test = bp.where_sunward('saturn_main_rings')
    show_info('Mask of rings sunward', test)

    test = bp.where_antisunward('saturn_main_rings')
    show_info('Mask of rings anti-sunward', test)

    ########################

    if printing: print('\n********* borders')

    mask = bp.where_intercepted('saturn')
    test = bp.border_inside(mask)
    show_info('Border of Saturn intercepted mask, inside', test)

    test = bp.border_outside(mask)
    show_info('Border of Saturn intercepted mask, outside', test)

    test = bp.border_below(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius below 100 kkm', test)

    test = bp.border_atop(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius atop 100 kkm', test)

    test = bp.border_above(('ring_radius', 'saturn:ring'), 100.e3)
    show_info('Border of ring radius above 100 kkm', test)

    test = bp.evaluate(('border_above', ('ring_radius', 'saturn:ring'), 100.e3))
    show_info('Border of ring radius above 100 kkm via evaluate()', test)

    ########################

    if printing: print('\n********* EMPTY EVENTS')

    test = bp.where_below(('ring_radius', 'saturn_main_rings'), 10.e3)
    show_info('Empty mask of Saturn ring radius below 10 kkm', test)

    test = bp.ring_radius('pluto:ring')
    show_info('Empty ring radius for Pluto (km)', test)

    test = bp.longitude('pluto')
    show_info('Empty longitude for Pluto (km)', test)

    test = bp.incidence_angle('pluto')
    show_info('Empty incidence angle for Pluto (km)', test)

    config.LOGGING.off()

    return bp
