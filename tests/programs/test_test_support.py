##########################################################################################
# tests/programs/test_test_support.py: the resource paths of the gold master framework
##########################################################################################

import importlib.util
import os
import sys
from types import ModuleType

import pytest

from filecache import FCPath

# The module under test resolves every path at import time, so each case here loads a
# fresh copy under a private name rather than reloading the shared one. Reloading the
# shared module would replace the prefixes that the rest of the test suite is using.
MODULE_NAME = 'programs.gold_master.test_support'

# The environment variables the module reads
VARIABLES = ('OOPS_RESOURCES', 'OOPS_TEST_DATA_PATH', 'OOPS_GOLD_MASTER_PATH',
             'OOPS_BACKPLANE_OUTPUT_PATH')


def _load(monkeypatch: pytest.MonkeyPatch, **environment: str) -> ModuleType:
    """Import a private copy of the module with the given environment.

    Parameters:
        monkeypatch: The fixture used to set the environment variables.
        environment: The variables to define; every other one is removed.

    Returns:
        ModuleType: The freshly imported module.
    """

    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for (name, value) in environment.items():
        monkeypatch.setenv(name, value)

    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_the_resource_tree_defines_every_other_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """OOPS_RESOURCES alone fixes the test data, SPICE, and gold master locations."""

    module = _load(monkeypatch, OOPS_RESOURCES='/resources')

    assert module.OOPS_TEST_DATA_PATH == '/resources/test_data'
    assert module.OOPS_GOLD_MASTER_PATH == '/resources/gold_master'
    assert module.TEST_DATA_PREFIX == FCPath('/resources/test_data')
    assert module.TEST_SPICE_PREFIX == FCPath('/resources/test_data/SPICE')
    assert module.GOLD_MASTER_PREFIX == FCPath('/resources/gold_master')


def test_each_path_can_be_overridden_on_its_own(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable that is set replaces the location derived from the resource tree."""

    module = _load(monkeypatch, OOPS_RESOURCES='/resources',
                   OOPS_TEST_DATA_PATH='/elsewhere/data',
                   OOPS_GOLD_MASTER_PATH='/elsewhere/masters')

    assert module.OOPS_TEST_DATA_PATH == '/elsewhere/data'
    assert module.OOPS_GOLD_MASTER_PATH == '/elsewhere/masters'
    assert module.TEST_SPICE_PREFIX == FCPath('/elsewhere/data/SPICE')


def test_the_backplane_output_defaults_to_the_working_directory(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing set, generated backplanes are written where the tool is run."""

    module = _load(monkeypatch, OOPS_RESOURCES='/resources')

    assert module.OOPS_BACKPLANE_OUTPUT_PATH == os.getcwd()
    assert module.BACKPLANE_OUTPUT_PREFIX == FCPath(os.getcwd())


def test_the_backplane_output_can_be_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit output path replaces the working directory."""

    module = _load(monkeypatch, OOPS_RESOURCES='/resources',
                   OOPS_BACKPLANE_OUTPUT_PATH='/elsewhere/output')

    assert module.BACKPLANE_OUTPUT_PREFIX == FCPath('/elsewhere/output')


def test_every_path_is_undefined_without_the_resource_tree(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With no environment at all, the paths are None rather than wrong."""

    module = _load(monkeypatch)

    assert module.OOPS_RESOURCES is None
    assert module.OOPS_TEST_DATA_PATH is None
    assert module.OOPS_GOLD_MASTER_PATH is None
    assert module.TEST_DATA_PREFIX is None
    assert module.TEST_SPICE_PREFIX is None
    assert module.GOLD_MASTER_PREFIX is None


def test_a_cloud_resource_is_carried_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URI names a remote resource, which the file cache resolves the same way."""

    module = _load(monkeypatch, OOPS_RESOURCES='gs://rms-oops-resources')

    assert module.TEST_SPICE_PREFIX == FCPath('gs://rms-oops-resources/test_data/SPICE')
    assert module.GOLD_MASTER_PREFIX == FCPath('gs://rms-oops-resources/gold_master')


def test_the_shared_module_is_left_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading a private copy does not disturb the module the test suite is using."""

    before = sys.modules[MODULE_NAME].TEST_SPICE_PREFIX

    _load(monkeypatch, OOPS_RESOURCES='/resources')

    assert sys.modules[MODULE_NAME].TEST_SPICE_PREFIX is before

##########################################################################################
