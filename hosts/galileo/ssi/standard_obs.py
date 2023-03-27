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

###################################################################
name = 'C0349632100R'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name C0349632100R --preview
#  python gold_master.py --name C0349632100R --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'galileo/GO_0017/G1/GANYMEDE/' + name + '.img'),
        index   = 5,
        module  = 'hosts.galileo.ssi',
        planet  = '',
        moon    = 'GANYMEDE',
        ring    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

#gm.override('Celestial north minus east angles (deg)', 8., names=name)
#gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names=name)
#gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names=name)
#gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names=name)


