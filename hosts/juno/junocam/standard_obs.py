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

# Because JunoCam has such a large, distorted FOV, we need to assign the
# backplanes an especially large inventory border: border=10 seems to work.
# However, inventory=False is safer still.
gm.set_default_args(module='hosts.juno.junocam', inventory=False, border=10)

###################################################################
name = 'JNCR_2016347_03C00192_V01'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JNCR_2016347_03C00192_V01 --preview
#  python gold_master.py --name JNCR_2016347_03C00192_V01 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/junocam/03/' + name + '.img'),
        index   = 5,
        planets  = 'JUPITER',
        moons    = '',
        rings    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('Celestial north minus east angles (deg)', 8., names=name)
gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names=name)
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names=name)
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names=name)


####################################################################
#name = 'JNCR_2019096_19M00012_V02'
####################################################################
## To preview and adopt gold masters:
##  python gold_master.py --name JNCR_2019096_19M00012_V02 --preview
##  python gold_master.py --name JNCR_2019096_19M00012_V02 --adopt
#
#gm.define_standard_obs(name,
#        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                               'juno/junocam/19/' + name + '.img'),
#        index   = 7,
#        planets  = 'JUPITER',
#        moons    = '',
#        rings    = '',
#        kwargs  = {'snap':False, 'inventory':False, 'border':10})
#
#gm.override('Celestial north minus east angles (deg)', 8., names=name)
#gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names=name)
#gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names=name)
#gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names=name)


####################################################################
#name = 'JNCR_2019149_20G00008_V01'
####################################################################
## To preview and adopt gold masters:
##  python gold_master.py --name JNCR_2019149_20G00008_V01 --preview
##  python gold_master.py --name JNCR_2019149_20G00008_V01 --adopt
#
#gm.define_standard_obs(name,
#        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                               'juno/junocam/20/' + name + '.img'),
#        index   = 2,
#        planets  = 'JUPITER',
#        moons    = '',
#        rings    = '',
#        kwargs  = {'snap':False, 'inventory':False, 'border':10})
#
## this one has ansa problems because so high above equator 
#
#gm.override('Celestial north minus east angles (deg)', 8., names=name)
#gm.override('JUPITER longitude d/dv self-check (deg/pix)', 0.3, names=name)
#gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', 0.1, names=name)
#gm.override('JUPITER:RING emission angle, ring minus center (deg)', 8., names=name)



