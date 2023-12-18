################################################################################
# oops/hosts/cassini/standard_obs.py:
#
#  Standard gold-master observation definitions for Cassini ISS.
#
################################################################################
import os
import oops.backplane.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

# Define the default observation
gm.define_standard_obs('W1573721822_1',
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'cassini/ISS/W1573721822_1.IMG'),
        index   = None,
        planets = ['SATURN'],
        moons   = ['EPIMETHEUS'],
        rings   = ['SATURN_MAIN_RINGS'])

gm.set_default_args(module='oops.hosts.cassini.iss')

# The d/dv numerical ring derivatives are extra-uncertain due to the high
# foreshortening in the vertical direction.

gm.override('SATURN longitude d/du self-check (deg/pix)', 0.3)
gm.override('SATURN longitude d/dv self-check (deg/pix)', 0.05)
gm.override('SATURN_MAIN_RINGS azimuth d/dv self-check (deg/pix)', 1.)
gm.override('SATURN_MAIN_RINGS distance d/dv self-check (km/pix)', 0.3)
gm.override('SATURN_MAIN_RINGS longitude d/dv self-check (deg/pix)', 1.)
gm.override('SATURN:RING azimuth d/dv self-check (deg/pix)', 0.1)
gm.override('SATURN:RING distance d/dv self-check (km/pix)', 0.3)
gm.override('SATURN:RING longitude d/dv self-check (deg/pix)', 0.1)
