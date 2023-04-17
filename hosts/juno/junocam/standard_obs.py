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

####################################################################
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

gm.override('JUPITER:RING emission angle, ring minus center (deg)', 7., names=name)
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', .07, names=name)
gm.override('Celestial north minus east angles (deg)', 7., names=name)
gm.override('JUPITER longitude d/dv self-check (deg/pix)', .25, names=name)


###################################################################
name = 'JNCR_2020366_31C00065_V01'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JNCR_2020366_31C00065_V01 --preview
#  python gold_master.py --name JNCR_2020366_31C00065_V01 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/junocam/31/' + name + '.img'),
        index   = 12,
        planets  = 'JUPITER',
        moons    = '',
        rings    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('JUPITER:RING emission angle, ring minus center (deg)', 24., names=name)
gm.override('Celestial north minus east angles (deg)', 8., names=name)
gm.override('JUPITER longitude d/du self-check (deg/pix)', .4, names=name)
gm.override('JUPITER longitude d/dv self-check (deg/pix)', .3, names=name)


###################################################################
name = 'JNCR_2019096_19M00012_V02'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JNCR_2019096_19M00012_V02 --preview
#  python gold_master.py --name JNCR_2019096_19M00012_V02 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                               'juno/junocam/19/' + name + '.img'),
        index   = 7,
        planets  = 'JUPITER',
        moons    = '',
        rings    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('JUPITER:RING emission angle, ring minus center (deg)', 31., names=name)
gm.override('JUPITER:RING longitude d/dv self-check (deg/pix)', .09, names=name)
gm.override('JUPITER:RING azimuth d/dv self-check (deg/pix)', .03, names=name)
gm.override('Celestial north minus east angles (deg)', 9., names=name)
gm.override('JUPITER longitude d/du self-check (deg/pix)', .26, names=name)


###################################################################
name = 'JNCR_2019149_20G00008_V01'
###################################################################
# To preview and adopt gold masters:
#  python gold_master.py --name JNCR_2019149_20G00008_V01 --preview
#  python gold_master.py --name JNCR_2019149_20G00008_V01 --adopt

gm.define_standard_obs(name,
        obspath = os.path.join(TESTDATA_PARENT_DIRECTORY,
                           'juno/junocam/20/' + name + '.img'),
        index   = 2,
        planets  = 'JUPITER',
        moons    = '',
        rings    = '',
        kwargs  = {'snap':False, 'inventory':False, 'border':10})

gm.override('JUPITER:ANSA radius d/du self-check (km/pix)', 1.9, names=name)
gm.override('JUPITER:ANSA radius d/dv self-check (km/pix)', 1.8, names=name)
gm.override('JUPITER:ANSA altitude d/du self-check (km/pix)', 99., names=name)
gm.override('JUPITER:ANSA altitude d/dv self-check (km/pix)', 52., names=name)
gm.override('JUPITER:RING azimuth to observer, apparent minus actual (deg)', .18, names=name)
gm.override('JUPITER:RING emission angle, ring minus center (deg)', 41., names=name)
gm.override('JUPITER:RING longitude d/du self-check (deg/pix)', .04, names=name)
gm.override('JUPITER:RING longitude d/dv self-check (deg/pix)', .09, names=name)
gm.override('JUPITER:RING azimuth d/du self-check (deg/pix)', .16, names=name)
gm.override('JUPITER:RING azimuth d/dv self-check (deg/pix)', .23, names=name)
gm.override('Celestial north minus east angles (deg)', 6., names=name)




