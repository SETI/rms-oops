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
                   kwargs = {})

gm.set_default_args(inventory=True, border=4)

if __name__ == '__main__':
    gm.execute_as_command()

################################################################################
