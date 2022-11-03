################################################################################
# oops/backplane/exercise_backplanes.py
################################################################################

from __future__ import print_function

import numpy as np

from oops.backplane    import Backplane
from oops.event        import Event
from oops.meshgrid     import Meshgrid
from oops.surface.ansa import Ansa
from oops.constants    import HALFPI, DPR
import oops.config as config
from oops.backplane.unittester_support    import show_info


#===============================================================================
def exercise_backplanes(obs, printing, logging, saving, dir, refdir,
                        planet_key=None, moon_key=None, ring_key=None,
                        undersample=16, use_inventory=False, inventory_border=2):
    """Generates info from every backplane."""

    if printing and logging:
        config.LOGGING.on('        ')

    if printing:
        print()

    meshgrid = Meshgrid.for_fov(obs.fov, undersample=undersample, swap=True)

    if use_inventory:
        bp = Backplane(obs, meshgrid, inventory={})
    else:
        bp = Backplane(obs, meshgrid, inventory=None)



    if printing:
        print('\n********* right ascension')
    import oops.backplane.sky as sky
    sky.exercise_right_ascension(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* declination')
    sky.exercise_declination(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* celestial and polar angles')
    sky.exercise_celestial_and_polar_angles(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* pole angles')
    import oops.backplane.pole as pole
    pole.exercise(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* observer distances')
    import oops.backplane.distance as distance
    distance.exercise_observer(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* Sun distances')
    distance.exercise_sun(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* observer light time')
    distance.exercise_observer_light_time(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* Sun light time')
    distance.exercise_sun_light_time(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* event time')
    distance.exercise_event_time(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* planet/moon resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_surface(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring resolution')
    import oops.backplane.ring as ring
    ring.exercise_resolution(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ansa resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_ansa(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    import oops.backplane.ansa as ansa
    ansa.exercise_resolution(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_limb(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface latitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_surface_latitude(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_surface_planet_moon(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface incidence, emission, phase')
    import oops.backplane.lighting as lighting
    lighting.exercise_planet(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring radius, radial modes')
    import oops.backplane.ring as ring
    ring.exercise_radial_modes(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring longitude, azimuth')
    ring.exercise_radial_longitude_azimuth(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring phase angle')
    import oops.backplane.lighting as lighting
    lighting.exercise_ring(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring observer-sun longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_ring(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring incidence, solar elevation')
    import oops.backplane.ring as ring
    ring.exercise_photometry(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ansa geometry')
    import oops.backplane.ansa as ansa
    ansa.exercise_geometry(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb altitude')
    import oops.backplane.limb as limb
    limb.exercise(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_limb_longitude(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb latitude')
    spheroid.exercise_limb_latitude(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* orbit longitude')
    import oops.backplane.orbit as orbit
    orbit.exercise_longitude(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* masks')
    import oops.backplane.where as where
    where.exercise(bp, obs, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)






# Used for testing other images
#     test = bp.longitude('enceladus')
#     show_info('Enceladus longitude (deg)', test * DPR)
#
#     test = bp.sub_observer_longitude('enceladus')
#     show_info('Enceladus sub-observer longitude (deg)', test * DPR)
#
#     test = bp.sub_solar_longitude('enceladus')
#     show_info('Enceladus sub-solar longitude (deg)', test * DPR)


    config.LOGGING.off()

    return bp
