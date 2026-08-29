##########################################################################################
# tests/hosts/galileo/ssi/test_gold_master.py
##########################################################################################

import pytest

import programs.gold_master as gm

from oops.body import Body


# def test_AAA_Galileo_SSI_index_file():
#     dir = '/home/spitale/SETI/RMS/metadata/GO_0xxx/GO_0017'
# #    dir = f'{OOPS_TEST_DATA_PATH}/galileo/GO_0017'
#
#     obs = from_index(os.path.join(dir, 'GO_0017_index.lbl'),
#                      os.path.join(dir, 'GO_0017_supplemental_index.lbl'))


@pytest.fixture(autouse=True)
def _standard_obs():
    # Imported for its side effect: it defines the standard observations.
    from tests.hosts.galileo.ssi import standard_obs     # noqa: F401

    yield

    Body._undefine_solar_system()
    Body.define_solar_system()


def test_C0349632100R():
    gm.execute_as_pytest('C0349632100R')


def test_C0368369200R():
    gm.execute_as_pytest('C0368369200R')


def test_C0061455700R():
    gm.execute_as_pytest('C0061455700R')


def test_C0374685140R():
    gm.execute_as_pytest('C0374685140R')

##########################################################################################
