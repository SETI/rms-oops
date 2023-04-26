################################################################################
# hosts/galileo/ssi/standard_obs.py: 
#
#  Standard gold-master observation definitions for Galileo SSI.
#
################################################################################
import os
import unittest
import oops.backplane.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

gm.set_default_args(module='hosts.galileo.ssi', inventory=False, border=4)

###################################################################
name = 'C0349632100R'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name C0349632100R --preview
#  python gold_master.py --name C0349632100R --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'galileo/GO_0017/G1/GANYMEDE/' + name + '.img'),
        index    = None,
        planets  = '',
        moons    = 'GANYMEDE',
        rings    = '')

###################################################################
name = 'C0368369200R'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name C0368369200R --preview
#  python gold_master.py --name C0368369200R --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'galileo/GO_0017/C3/JUPITER/' + name + '.img'),
        index    = None,
        planets  = '',
        moons    = 'JUPITER',
        rings    = '')

gm.override('JUPITER:RING longitude d/du self-check (deg/pix)', .00025, names=name)
gm.override('JUPITER:RING longitude d/dv self-check (deg/pix)', .00014, names=name)
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', .0026, names=name)
gm.override('JUPITER:RING azimuth d/dv self-check (deg/pix)', .00026, names=name)
gm.override('JUPITER longitude d/dv self-check (deg/pix)', .096, names=name)

###################################################################
name = 'C0061455700R'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name C0061455700R --preview
#  python gold_master.py --name C0061455700R --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'galileo/GO_0004/EARTH/' + name + '.img'),
        index    = None,
        planets  = '',
        moons    = 'EARTH',
        rings    = '')

###################################################################
name = 'C0374685140R'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name C0374685140R --preview
#  python gold_master.py --name C0374685140R --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'galileo/GO_0017/E4/EUROPA/' + name + '.img'),
        index    = None,
        planets  = '',
        moons    = 'EUROPA',
        rings    = '')
