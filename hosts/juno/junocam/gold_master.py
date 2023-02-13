################################################################################
# hosts/juno/junocam/gold_master.py: Backplane gold master tester for JunoCam
################################################################################

import os

import oops.backplane.gold_master as gm
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                    'juno/junocam/03/JNCR_2016347_03C00192_V01.img')

gm.set_default_obs(obspath = obspath,
                   module = 'hosts.juno.junocam',
                   index  = 5,
                   planet = 'JUPITER',
                   moon   = [],
                   ring   = [],
                   kwargs = {'snap': False})

gm.override('Celestial north minus east angles (deg)', 8.)
gm.override('JUPITER center resolution along u axis (km)', 0.1)
gm.override('JUPITER center resolution along v axis (km)', 0.1)
gm.override('JUPITER:ANSA center resolution along u axis (km)', 0.1)
gm.override('JUPITER:ANSA center resolution along v axis (km)', 0.1)
gm.override('JUPITER:LIMB center resolution along u axis (km)', 0.1)
gm.override('JUPITER:LIMB center resolution along v axis (km)', 0.1)
gm.override('JUPITER:RING azimuth minus longitude wrt Sun (deg)', None)
gm.override('JUPITER:RING center resolution along u axis (km)', 0.1)
gm.override('JUPITER:RING center resolution along v axis (km)', 0.1)
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 40.)
gm.override('JUPITER:RING incidence angle, ring minus center (deg)', 3.)

gm.set_default_args(inventory=True, border=4)

if __name__ == '__main__':
    gm.execute_as_command()

################################################################################
