################################################################################
# hosts/juno/junocam/standard_obs.py: 
#
#  Standard gold-master observation definitions for JunoCam.
#
################################################################################
import os
import unittest
import oops.backplane.gold_master as gm

from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

###############################
# JNCR_2016347_03C00192_V01
###############################
gm.define_standard_obs('JNCR_2016347_03C00192_V01',
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/junocam/03/'
                               'JNCR_2016347_03C00192_V01.img'),
        index   = 5,
        module  = 'hosts.juno.junocam',
        planet  = 'JUPITER',
        moon    = '',
        ring    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('Celestial north minus east angles (deg)', 8., names='JNCR_2016347_03C00192_V01')
gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names='JNCR_2016347_03C00192_V01')
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names='JNCR_2016347_03C00192_V01')
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names='JNCR_2016347_03C00192_V01')

###############################
# JNCR_2019096_19M00012_V02
###############################
gm.define_standard_obs('JNCR_2019096_19M00012_V02',
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/junocam/19/'
                               'JNCR_2019096_19M00012_V02.img'),
        index   = 7,
        module  = 'hosts.juno.junocam',
        planet  = 'JUPITER',
        moon    = '',
        ring    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('Celestial north minus east angles (deg)', 8., names='JNCR_2019096_19M00012_V02')
gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names='JNCR_2019096_19M00012_V02')
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names='JNCR_2019096_19M00012_V02')
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names='JNCR_2019096_19M00012_V02')


