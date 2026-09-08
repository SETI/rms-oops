##########################################################################################
# tests/programs/test_gold_master.py: the registration API of the gold master framework
##########################################################################################

import copy
from collections.abc import Iterator

import pytest

import programs.gold_master as gm
import programs.gold_master.all             # noqa: F401  (registers the test suites)


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Save and restore the module-level registries these tests modify.

    The gold master framework keeps its configuration in module-level dictionaries, so
    a test that registers an observation would otherwise leak into every test that runs
    afterward, including the host tests.
    """

    saved_obs = copy.deepcopy(gm.STANDARD_OBS_INFO)
    saved_defaults = copy.deepcopy(gm.DEFAULTS)
    saved_overrides = copy.deepcopy(dict(gm.TEST_OVERRIDES))
    saved_suites = dict(gm.TEST_SUITES)

    yield

    gm.STANDARD_OBS_INFO.clear()
    gm.STANDARD_OBS_INFO.update(saved_obs)
    gm.DEFAULTS.clear()
    gm.DEFAULTS.update(saved_defaults)
    gm.TEST_OVERRIDES.clear()
    gm.TEST_OVERRIDES.update(saved_overrides)
    gm.TEST_SUITES.clear()
    gm.TEST_SUITES.update(saved_suites)


##########################################################################################
# module_dirname
##########################################################################################

@pytest.mark.parametrize('module, expected',
                         [('oops.hosts.cassini.iss', 'cassini.iss'),
                          ('cassini.iss', 'cassini.iss'),
                          ('tests.hosts.galileo.ssi', 'galileo.ssi'),
                          ('a.b.c.d.e', 'd.e')])
def test_module_dirname_keeps_the_last_two_components(module: str,
                                                      expected: str) -> None:
    """The directory is named for the mission and the instrument alone."""

    assert gm.module_dirname(module) == expected


@pytest.mark.parametrize('module', ['iss', ''])
def test_a_short_module_name_is_its_own_directory(module: str) -> None:
    """A name with fewer than two components is used as it is."""

    assert gm.module_dirname(module) == module


##########################################################################################
# define_standard_obs and set_default_obs
##########################################################################################

def test_define_standard_obs_records_the_observation() -> None:
    """A standard observation is registered under the name it is given."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', 0, planets='SATURN')

    assert 'test_obs' in gm.STANDARD_OBS_INFO
    assert gm.STANDARD_OBS_INFO['test_obs']['obspath'] == '/data/test_obs.img'
    assert gm.STANDARD_OBS_INFO['test_obs']['index'] == 0


def test_a_single_body_name_becomes_a_sequence() -> None:
    """The planets, moons, and rings are stored as sequences of names."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', 0,
                           planets='SATURN', moons='MIMAS', rings='A_RING')
    info = gm.STANDARD_OBS_INFO['test_obs']

    assert info['planets'] == ('SATURN',)
    assert info['moons'] == ('MIMAS',)
    assert info['rings'] == ('A_RING',)


def test_a_list_of_body_names_is_kept() -> None:
    """Several planets or moons can be named at once."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', 0,
                           planets=['SATURN', 'JUPITER'], moons=['MIMAS', 'DIONE'])
    info = gm.STANDARD_OBS_INFO['test_obs']

    assert list(info['planets']) == ['SATURN', 'JUPITER']
    assert list(info['moons']) == ['MIMAS', 'DIONE']


def test_the_bodies_default_to_empty() -> None:
    """Moons and rings are optional."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', 0, planets='SATURN')
    info = gm.STANDARD_OBS_INFO['test_obs']

    assert info['moons'] == ()
    assert info['rings'] == ()


def test_keyword_arguments_are_recorded() -> None:
    """Any keyword arguments for from_file are stored with the observation."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', 0, planets='SATURN',
                           kwargs={'fast_distortion': True})

    assert gm.STANDARD_OBS_INFO['test_obs']['kwargs'] == {'fast_distortion': True}


def test_an_index_of_none_means_every_observation() -> None:
    """A file yielding several observations can be tested in full."""

    gm.define_standard_obs('test_obs', '/data/test_obs.img', None, planets='SATURN')

    assert gm.STANDARD_OBS_INFO['test_obs']['index'] is None


def test_set_default_obs_registers_under_the_default_name() -> None:
    """The default observation is the one used when no name is given."""

    gm.set_default_obs('/data/default.img', None, 'JUPITER')

    assert gm.STANDARD_OBS_INFO['default']['obspath'] == '/data/default.img'
    assert gm.STANDARD_OBS_INFO['default']['planets'] == ('JUPITER',)


def test_a_second_definition_replaces_the_first() -> None:
    """Re-registering a name overwrites the earlier entry."""

    gm.define_standard_obs('test_obs', '/data/first.img', 0, planets='SATURN')
    gm.define_standard_obs('test_obs', '/data/second.img', 1, planets='JUPITER')

    assert gm.STANDARD_OBS_INFO['test_obs']['obspath'] == '/data/second.img'
    assert gm.STANDARD_OBS_INFO['test_obs']['index'] == 1


##########################################################################################
# set_default_args
##########################################################################################

def test_set_default_args_replaces_a_default() -> None:
    """Each option given replaces the matching default for subsequent tests."""

    gm.set_default_args(tolerance=2., undersample=8)

    assert gm.DEFAULTS['tolerance'] == 2.
    assert gm.DEFAULTS['undersample'] == 8


def test_set_default_args_leaves_the_other_defaults_alone() -> None:
    """Only the named options change."""

    before = gm.DEFAULTS['radius']
    gm.set_default_args(tolerance=2.)

    assert gm.DEFAULTS['radius'] == before


def test_an_unrecognized_option_is_stored_but_unused() -> None:
    """A keyword the framework does not know is kept without effect."""

    gm.set_default_args(not_a_real_option=7)

    assert gm.DEFAULTS['not_a_real_option'] == 7


def test_the_default_task_is_to_compare() -> None:
    """A run compares against the gold masters unless told otherwise."""

    assert gm.DEFAULTS['task'] == 'compare'


def test_missing_gold_masters_are_an_error_by_default() -> None:
    """ignore_missing defaults to False, so a missing master fails the run."""

    assert gm.DEFAULTS['ignore_missing'] is False


##########################################################################################
# override
##########################################################################################

def test_override_applies_to_every_standard_observation() -> None:
    """With no names given, the override reaches all of them."""

    gm.define_standard_obs('obs_a', '/data/a.img', 0, planets='SATURN')
    gm.define_standard_obs('obs_b', '/data/b.img', 0, planets='SATURN')

    gm.override('A TEST TITLE', 1.5)

    assert gm.TEST_OVERRIDES['obs_a']['A TEST TITLE'] == 1.5
    assert gm.TEST_OVERRIDES['obs_b']['A TEST TITLE'] == 1.5


def test_override_can_name_one_observation() -> None:
    """A single name limits the override to that observation."""

    gm.define_standard_obs('obs_a', '/data/a.img', 0, planets='SATURN')
    gm.define_standard_obs('obs_b', '/data/b.img', 0, planets='SATURN')

    gm.override('A TEST TITLE', 1.5, names='obs_a')

    assert gm.TEST_OVERRIDES['obs_a']['A TEST TITLE'] == 1.5
    assert 'A TEST TITLE' not in gm.TEST_OVERRIDES['obs_b']


def test_override_can_name_several_observations() -> None:
    """A list of names limits the override to those observations."""

    for name in ('obs_a', 'obs_b', 'obs_c'):
        gm.define_standard_obs(name, f'/data/{name}.img', 0, planets='SATURN')

    gm.override('A TEST TITLE', 1.5, names=['obs_a', 'obs_c'])

    assert 'A TEST TITLE' in gm.TEST_OVERRIDES['obs_a']
    assert 'A TEST TITLE' not in gm.TEST_OVERRIDES['obs_b']
    assert 'A TEST TITLE' in gm.TEST_OVERRIDES['obs_c']


def test_a_value_of_none_suppresses_the_test() -> None:
    """None means the test is skipped rather than compared."""

    gm.define_standard_obs('obs_a', '/data/a.img', 0, planets='SATURN')
    gm.override('A TEST TITLE', None, names='obs_a')

    assert gm.TEST_OVERRIDES['obs_a']['A TEST TITLE'] is None


##########################################################################################
# The test suite registry
##########################################################################################

def test_every_backplane_family_registers_a_test_suite() -> None:
    """Importing `all` registers one suite per backplane module."""

    expected = {'ansa', 'border', 'distance', 'lighting', 'limb', 'orbit', 'pole',
                'resolution', 'ring', 'sky', 'spheroid', 'where'}

    assert expected <= set(gm.TEST_SUITES)


def test_get_test_suite_returns_the_registered_function() -> None:
    """A registered suite is looked up by its name."""

    def _suite(bpt: object) -> None:
        """A test suite that does nothing."""

    gm.register_test_suite('test_only_suite', _suite)

    assert gm.get_test_suite('test_only_suite') is _suite


def test_get_test_suite_rejects_an_unknown_name() -> None:
    """A name that was never registered raises KeyError."""

    with pytest.raises(KeyError):
        gm.get_test_suite('not_a_registered_suite')


def test_registering_a_suite_twice_replaces_it() -> None:
    """The most recent registration wins."""

    def _first(bpt: object) -> None:
        """The first suite."""

    def _second(bpt: object) -> None:
        """The replacement suite."""

    gm.register_test_suite('test_only_suite', _first)
    gm.register_test_suite('test_only_suite', _second)

    assert gm.get_test_suite('test_only_suite') is _second

##########################################################################################
