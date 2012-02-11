################################################################################
# oops/tools.py
#
# 2/6/12 Created (MRS) - Based on parts of oops.py.
################################################################################

import numpy as np
import os
import spicedb
import julian

################################################################################
# Useful class methods to define SPICE-related quantities
################################################################################

LSK_LOADED = False

def load_leap_seconds():
    """Loads the most recent leap seconds kernel if it was not already loaded.
    """

    global LSK_LOADED

    if LSK_LOADED: return

    # Query for the most recent LSK
    spicedb.open_db()
    lsk = spicedb.select_kernels("LSK")
    spicedb.close_db()

    # Furnish the LSK to the SPICE toolkit
    spicedb.furnish_kernels(lsk)

    # Initialize the Julian toolkit
    julian.load_from_kernel(os.path.join(spicedb.get_spice_path(),
                                         lsk[0].filespec))

    LSK_LOADED = True

def define_solar_system(start_time, stop_time, asof=None):
    """Constructs SpicePaths for all the planets and moons in the solar system
    (including Pluto). Each planet is defined relative to the SSB. Each moon
    is defined relative to its planet. Names are as defined within the SPICE
    toolkit.

    Input:
        start_time      start_time of the period to be convered, in ISO date or
                        date-time format.
        stop_time       stop_time of the period to be covered, in ISO date or
                        date-time format.
        asof            a UTC date such that only kernels released earlier than
                        that date will be included, in ISO format.
    """

    # Always load the most recent Leap Seconds kernel, but only once
    load_leap_seconds()

    # Convert the formats to times as recognized by spicedb.
    (day, sec) = julian.day_sec_from_iso(start_time)
    start_time = julian.ymdhms_format_from_day_sec(day, sec)

    (day, sec) = julian.day_sec_from_iso(stop_time)
    stop_time = julian.ymdhms_format_from_day_sec(day, sec)

    if asof is not None:
        (day, sec) = julian.day_sec_from_iso(stop_time)
        asof = julian.ymdhms_format_from_day_sec(day, sec)

    # Load the necessary SPICE kernels
    spicedb.open_db()
    spicedb.furnish_solar_system(start_time, stop_time, asof)
    spicedb.close_db()

    # Sun...
    ignore = oops.SpicePath("SUN","SSB")

    # Mercury...
    define_planet("MERCURY", [], [])

    # Venus...
    define_planet("VENUS", [], [])

    # Earth...
    define_planet("EARTH", ["MOON"], [])

    # Mars...
    define_planet("MARS", spicedb.MARS_ALL_MOONS, [])

    # Jupiter...
    define_planet("JUPITER", spicedb.JUPITER_REGULAR,
                             spicedb.JUPITER_IRREGULAR)

    # Saturn...
    define_planet("SATURN", spicedb.SATURN_REGULAR,
                            spicedb.SATURN_IRREGULAR)

    # Uranus...
    define_planet("URANUS", spicedb.URANUS_REGULAR,
                            spicedb.URANUS_IRREGULAR)

    # Neptune...
    define_planet("NEPTUNE", spicedb.NEPTUNE_REGULAR,
                             spicedb.NEPTUNE_IRREGULAR)

    # Pluto...
    define_planet("PLUTO", spicedb.PLUTO_ALL_MOONS, [])

def define_planet(planet, regular_ids, irregular_ids):

    # Define the planet's path and frame
    ignore = oops.SpicePath(planet, "SSB")
    ignore = oops.SpiceFrame(planet)

    # Define the SpicePaths of individual regular moons
    regulars = []
    for id in regular_ids:
        regulars += [oops.SpicePath(id, planet)]
        try:        # Some frames for small moons are not defined
            ignore = oops.SpiceFrame(id)
        except RuntimeError: pass
        except LookupError: pass

    # Define the Multipath of the regular moons, with and without the planet
    ignore = oops.MultiPath(regulars, planet,
                            id=(planet + "_REGULARS"))
    ignore = oops.MultiPath([planet] + regulars, "SSB",
                            id=(planet + "+REGULARS"))

    # Without irregulars, we're just about done
    if irregular_ids == []:
        ignore = oops.MultiPath([planet] + regulars, "SSB",
                                 id=(planet + "+MOONS"))
        return

    # Define the SpicePaths of individual irregular moons
    irregulars = []
    for id in irregular_ids:
        irregulars += [oops.SpicePath(id, planet)]
        try:        # Some frames for small moons are not defined
            ignore = oops.SpiceFrame(id)
        except RuntimeError: pass
        except LookupError: pass

    # Define the Multipath of all the irregular moons
    ignore = oops.MultiPath(regulars, planet, id=(planet + "_IRREGULARS"))

    # Define the Multipath of all the moons, with and without the planet
    ignore = oops.MultiPath(regulars + irregulars, planet,
                            id=(planet + "_MOONS"))
    ignore = oops.MultiPath([planet] + regulars + irregulars, "SSB",
                            id=(planet + "+MOONS"))

################################################################################
