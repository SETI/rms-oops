################################################################################
# hosts/juno/jiram/standard_obs.py: 
#
#  Standard gold-master observation definitions for Juno JIRAM.
#
################################################################################
import os
import unittest
import oops.backplane.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

gm.set_default_args(module='hosts.juno.jiram', inventory=False, border=4)

####################################################################
name = 'JIR_IMG_RDR_2013282T133843_V03'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JIR_IMG_RDR_2013282T133843_V03 --preview
#  python gold_master.py --name JIR_IMG_RDR_2013282T133843_V03 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/jiram/JNOJIR_2000/DATA/' + name + '.IMG'),
        index   = 1,
        planets  = 'MOON',
        moons    = '',
        rings    = '')

gm.override('Celestial north minus east angles (deg)', None, names=name)

####################################################################
name = 'JIR_IMG_RDR_2017244T104633_V01'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JIR_IMG_RDR_2017244T104633_V01 --preview
#  python gold_master.py --name JIR_IMG_RDR_2017244T104633_V01 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/jiram/JNOJIR_2008/DATA/' + name + '.IMG'),
        index   = 1,
        planets  = 'EUROPA',
        moons    = '',
        rings    = '')

####################################################################
name = 'JIR_IMG_RDR_2018197T055537_V01'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JIR_IMG_RDR_2018197T055537_V01 --preview
#  python gold_master.py --name JIR_IMG_RDR_2018197T055537_V01 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/jiram/JNOJIR_2014/DATA/' + name + '.IMG'),
        index   = 0,
        planets  = 'JUPITER',
        moons    = '',
        rings    = '')

####################################################################
name = 'JIR_SPE_RDR_2013282T133845_V03'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JIR_SPE_RDR_2013282T133845_V03 --preview
#  python gold_master.py --name JIR_SPE_RDR_2013282T133845_V03 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/jiram/JNOJIR_2000/DATA/' + name + '.DAT'),
        index   = 0,
        planets  = 'MOON',
        moons    = '',
        rings    = '')



