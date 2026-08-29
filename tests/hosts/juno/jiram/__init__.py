##########################################################################################
# tests/hosts/juno/jiram/__init__.py
##########################################################################################

import pytest
import programs.gold_master as gm
import oops.hosts.juno.jiram as jiram

from oops.unittester_support import TEST_DATA_PREFIX


@pytest.fixture(autouse=True)
def _standard_obs():
    # JIRAM has a distorted FOV, so the backplanes need a generous inventory
    # border; inventory=False is safer still. These are run options, not
    # arguments to from_file, so they belong in set_default_args().
    gm.set_default_args(module='oops.hosts.juno.jiram', inventory=False,
                        border=4)

    gm.define_standard_obs('JIR_IMG_RDR_2013282T133843_V03',
        obspath = 'juno/jiram/JNOJIR_2000/DATA/JIR_IMG_RDR_2013282T133843_V03.IMG',
        index   = 1,
        planets = '',
        moons   = 'MOON',
        rings   = '')

    gm.define_standard_obs('JIR_IMG_RDR_2017244T104633_V01',
        obspath = 'juno/jiram/JNOJIR_2008/DATA/JIR_IMG_RDR_2017244T104633_V01.IMG',
        index   = 1,
        planets = '',
        moons   = 'EUROPA',
        rings   = '')

    gm.define_standard_obs('JIR_IMG_RDR_2018197T055537_V01',
        obspath = 'juno/jiram/JNOJIR_2014/DATA/JIR_IMG_RDR_2018197T055537_V01.IMG',
        index   = 0,
        planets = 'JUPITER',
        moons   = '',
        rings   = '')

    gm.define_standard_obs('JIR_SPE_RDR_2013282T133845_V03',
        obspath = 'juno/jiram/JNOJIR_2000/DATA/JIR_SPE_RDR_2013282T133845_V03.DAT',
        index   = 0,
        planets = '',
        moons   = 'MOON',
        rings   = '')


def test_1():
    gm.execute_as_pytest('JIR_IMG_RDR_2013282T133843_V03')


def test_2():
    gm.execute_as_pytest('JIR_IMG_RDR_2017244T104633_V01')


def test_3():
    gm.execute_as_pytest('JIR_IMG_RDR_2018197T055537_V01')


def test_4():
    gm.execute_as_pytest('JIR_SPE_RDR_2013282T133845_V03')
##########################################################################################
