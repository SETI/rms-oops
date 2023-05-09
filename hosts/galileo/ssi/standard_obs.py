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

gm.override('Right ascension d/dv self-check (deg/pix)', 2.2e-9, names=name)

# overrides to cover unexplained discrepancies among Mark, Rob, Joe
gm.override('GANYMEDE center distance from Sun (km)', 2, names=name)
gm.override('GANYMEDE center light time from Sun (km)', 6e-6, names=name)
gm.override('GANYMEDE center right ascension (deg, actual)', 3e-5, names=name)
gm.override('GANYMEDE center right ascension (deg, apparent)', 3e-5, names=name)
gm.override('GANYMEDE center declination (deg, actual)', 2e-4, names=name)
gm.override('GANYMEDE center declination (deg, apparent)', 2e-4, names=name)
gm.override('GANYMEDE center distance to observer (km)', 3., names=name)
gm.override('GANYMEDE center light time to observer (km)', 9e-6, names=name)



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

# overrides to cover unexplained discrepancies among Mark, Rob, Joe
gm.override('JUPITER center distance from Sun (km)', 2.5, names=name)
gm.override('JUPITER center light time from Sun (km)', 4e-6, names=name)
gm.override('JUPITER:RING center distance from Sun (km)', 2.5, names=name)
gm.override('JUPITER:RING center light time from Sun (km)', 9e-6, names=name)
gm.override('JUPITER center right ascension (deg, actual)', 1.5e-4, names=name)
gm.override('JUPITER center right ascension (deg, apparent)', 1.5e-4, names=name)
gm.override('JUPITER center declination (deg, actual)', 5e-4, names=name)
gm.override('JUPITER center declination (deg, apparent)', 5e-4, names=name)

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

# overrides to cover unexplained discrepancies among Mark, Rob, Joe
gm.override('EUROPA center distance from Sun (km)', 5., names=name)
gm.override('EUROPA center light time from Sun (km)', 1.5e-5, names=name)
gm.override('EUROPA pole clock angle (deg)', 0.0017, names=name)
gm.override('EUROPA pole position angle (deg)', 0.0017, names=name)
gm.override('EUROPA center right ascension (deg, actual)', 0.007, names=name)
gm.override('EUROPA center right ascension (deg, apparent)', 0.007, names=name)
gm.override('EUROPA center declination (deg, actual)', 0.017, names=name)
gm.override('EUROPA center declination (deg, apparent)', 0.017, names=name)
gm.override('EUROPA sub-observer latitude, planetocentric (deg)', 0.019, names=name)
gm.override('EUROPA sub-observer latitude, planetographic (deg)', 0.019, names=name)

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
