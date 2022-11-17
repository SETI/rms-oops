################################################################################
# oops/backplane/exercise_backplanes.py
################################################################################

from __future__ import print_function

import os

from oops.backplane    import Backplane
from oops.meshgrid     import Meshgrid
import oops.config as config

from oops.unittester_support            import TESTDATA_PARENT_DIRECTORY
from oops.backplane.unittester_support  import Backplane_Settings


#===============================================================================
def _exercise_backplanes(obs, printing, logging, saving, dir, refdir,
                         planet_key=None, moon_key=None, ring_key=None,
                         undersample=16, use_inventory=False,
                         inventory_border=2):
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
    sky.exercise_right_ascension(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* declination')
    sky.exercise_declination(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* celestial and polar angles')
    sky.exercise_celestial_and_polar_angles(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* pole angles')
    import oops.backplane.pole as pole
    pole.exercise(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* observer distances')
    import oops.backplane.distance as distance
    distance.exercise_observer(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* Sun distances')
    distance.exercise_sun(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* observer light time')
    distance.exercise_observer_light_time(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* Sun light time')
    distance.exercise_sun_light_time(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* event time')
    distance.exercise_event_time(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* planet/moon resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_surface(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring resolution')
    import oops.backplane.ring as ring
    ring.exercise_resolution(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ansa resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_ansa(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    import oops.backplane.ansa as ansa
    ansa.exercise_resolution(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb resolution')
    import oops.backplane.resolution as resolution
    resolution.exercise_limb(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface latitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_surface_latitude(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_surface_planet_moon(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* surface incidence, emission, phase')
    import oops.backplane.lighting as lighting
    lighting.exercise_planet(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring radius, radial modes')
    import oops.backplane.ring as ring
    ring.exercise_radial_modes(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring longitude, azimuth')
    ring.exercise_radial_longitude_azimuth(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring phase angle')
    import oops.backplane.lighting as lighting
    lighting.exercise_ring(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring observer-sun longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_ring(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ring incidence, solar elevation')
    import oops.backplane.ring as ring
    ring.exercise_photometry(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* ansa geometry')
    import oops.backplane.ansa as ansa
    ansa.exercise_geometry(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb altitude')
    import oops.backplane.limb as limb
    limb.exercise(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb longitude')
    import oops.backplane.spheroid as spheroid
    spheroid.exercise_limb_longitude(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* limb latitude')
    spheroid.exercise_limb_latitude(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* orbit longitude')
    import oops.backplane.orbit as orbit
    orbit.exercise_longitude(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

    if printing:
        print('\n********* masks')
    import oops.backplane.where as where
    where.exercise(bp, printing, saving, dir, refdir,
                               planet=planet_key, moon=moon_key, ring=ring_key)

# Used for testing other images
#     test = bp.longitude('enceladus')
#     show_info(bp, 'Enceladus longitude (deg)', test * DPR)
#
#     test = bp.sub_observer_longitude('enceladus')
#     show_info(bp, 'Enceladus sub-observer longitude (deg)', test * DPR)
#
#     test = bp.sub_solar_longitude('enceladus')
#     show_info(bp, 'Enceladus sub-solar longitude (deg)', test * DPR)


    config.LOGGING.off()



#===============================================================================
def exercise_backplanes_settings(obs):
    """Wrapper for exercise_backplanes(). Determines default directories
       based on input observation object and fills in args based on
       settings."""

    # determine default output and reference directories
    specname = obs.filespec.split( \
                    os.path.dirname(TESTDATA_PARENT_DIRECTORY) + '/')[1]
    basedir = os.path.dirname(specname)
    outdir = os.path.join(TESTDATA_PARENT_DIRECTORY,
                          'backplane_exercises', basedir)
    Backplane_Settings.REFERENCE = \
        os.path.join(outdir, 'reference_' + str(Backplane_Settings.UNDERSAMPLE))

    # determine specific output directory
    if Backplane_Settings.OUTPUT is None:
        Backplane_Settings.OUTPUT = outdir

    if Backplane_Settings.REF:
        Backplane_Settings.OUTPUT = Backplane_Settings.REFERENCE

    # Implement NO_COMPARE
    if Backplane_Settings.NO_COMPARE:
        Backplane_Settings.REFERENCE = None



#===============================================================================
def exercise_backplanes(obs, **kwargs):

    """Wrapper for exercise_backplanes(). Determines default directories
       based on input observation object and fills in args based on
       settings."""

    # complete settings that require observation information
    exercise_backplanes_settings(obs)


    # perform the exercises
    _exercise_backplanes(obs, Backplane_Settings.PRINTING,
                              Backplane_Settings.LOGGING,
                              Backplane_Settings.SAVING,
                              Backplane_Settings.OUTPUT,
                              Backplane_Settings.REFERENCE,
                              undersample=Backplane_Settings.UNDERSAMPLE,
                              **kwargs)