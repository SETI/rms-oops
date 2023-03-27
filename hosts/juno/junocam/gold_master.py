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
gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3)
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1)
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8.)

# Because JunoCam has such a large, distorted FOV, we need to assign the
# backplanes an especially large inventory border: border=10 seems to work.
# However, inventory=False is safer still.
gm.set_default_args(inventory=False, border=10)

if __name__ == '__main__':
    gm.execute_as_command()

################################################################################
