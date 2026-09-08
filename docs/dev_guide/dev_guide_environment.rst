Development environment
=======================

Setting up a checkout
---------------------

Clone the repository and let the setup script create the virtual environment. It refuses
an interpreter below Python 3.11, upgrades ``pip``, and installs the package in editable
mode with the ``dev`` extras, which pull in the ``docs`` extras as well::

    git clone https://github.com/SETI/rms-oops.git
    cd rms-oops
    ./scripts/setup-venv.sh
    source venv/bin/activate

``-r`` recreates an existing environment and ``-p CMD`` selects the interpreter. Never
install into the system Python: the packages live under ``src``, so nothing imports from
a bare checkout, and the check script and the tests both assume ``venv/``.

Environment variables
---------------------

Nothing runs without the external resource tree: the SPICE kernels, the test data and
the gold masters. ``OOPS_RESOURCES`` names it, and the rest default to subdirectories:

=============================== ==================================================
Variable                        Default
=============================== ==================================================
``OOPS_RESOURCES``              (required)
``SPICE_PATH``                  ``$OOPS_RESOURCES/SPICE``
``SPICE_SQLITE_DB_NAME``        ``$SPICE_PATH/SPICE.db``
``OOPS_TEST_DATA_PATH``         ``$OOPS_RESOURCES/test_data``
``OOPS_GOLD_MASTER_PATH``       ``$OOPS_RESOURCES/gold_master``
``OOPS_BACKPLANE_OUTPUT_PATH``  the current directory
``HST_IDC_PATH``                ``$OOPS_RESOURCES/HST/IDC``
``HST_SYN_PATH``                ``$OOPS_RESOURCES/HST/SYN``
=============================== ==================================================

Any of them may name a cloud resource such as ``gs://rms-oops-resources/gold_master``;
:mod:`programs.gold_master.test_support` turns each into a ``filecache`` prefix. A test
that fails on a missing kernel or a missing gold master when these are unset is an
environment problem, not a code defect. The documentation build is the exception: it
mocks ``cspyce`` and needs no resources.

Smoke test
----------

With the environment active and ``OOPS_RESOURCES`` set, this confirms that the kernels
are reachable and the geometry core works:

.. code-block:: python

    import oops
    from oops.body import Body

    Body.define_solar_system('2007-11-01', '2007-11-30', planets=6)
    print(Body.lookup('SATURN').surface)

The :doc:`User's Guide examples </user_guide/user_guide_examples>` go on from there.

Running the tests
-----------------

There are three suites, run as three invocations so that a failure is attributable to
one of them::

    pytest tests --ignore=tests/hosts --ignore=tests/spicedb   # the main suite
    pytest tests/hosts                                          # the gold master tests
    pytest tests/spicedb                                        # the spicedb tests

``pytest tests`` runs all three together and passes, but the check script and CI keep
them apart. A single module or test is selected as usual::

    pytest tests/frame/test_spinframe.py
    pytest tests/backplane -k ring_radius

Points to know before running them:

* ``pyproject.toml`` sets ``--cov`` in ``addopts``, so every run measures coverage over
  ``src`` and ``programs``; add ``--cov-report=term-missing`` to see the missing lines.
* ``-n auto`` is deliberately not configured. The gold master tests share the
  :class:`~oops.Body` registry and the ``QuickPath`` and ``QuickFrame`` caches with
  whatever else lands on the same worker, and one Galileo SSI test fails under that
  sharing. The comment in ``[tool.pytest.ini_options]`` records the reason.
* ``filterwarnings = ["error"]`` is likewise absent, because the spicedb tests and the
  legacy modules still raise warnings that have not been cleaned up.
* ``--gold-master DIR`` points the host tests at a different set of masters for one run;
  see :doc:`dev_guide_testing`.
* Only files named ``test_*.py`` are collected. Several host directories hold converted
  tests under other names, which no suite ran before the conversion either; wiring them
  up means fixing the tests, not renaming the files.

Running the checks
------------------

``scripts/run-all-checks.sh`` is the single source of truth for the quality gates. With no
arguments it runs everything in parallel; ``-s`` runs sequentially, which is easier to read
when something fails, and each gate has its own flag (``--ruff-check``, ``--mypy``,
``--pytest-hosts``, ``--sphinx`` and so on; ``-h`` lists them). It prints a summary naming
each gate that failed and exits non-zero if any did.

================ =====================================================================
Gate             Command
================ =====================================================================
ruff             ``ruff check .`` over the whole repository; the linter of record.
flake8           ``flake8 --select=E12,E13 src programs tests``; the continuation-line
                 indent checks alone, which ruff does not implement. ``.flake8`` holds
                 only that selection.
mypy             ``mypy``, over ``tests`` only. Under ``src`` the types live in the
                 docstrings, so checking it would report their absence, not defects.
stubtest         ``mypy.stubtest oops spicedb programs --allowlist stubtest-allowlist.txt
                 --ignore-missing-stub``; the ``.pyi`` stubs against the run-time API.
pyroma           ``pyroma --min=10 .``; the packaging metadata.
bandit           ``bandit -q -c pyproject.toml -r src``.
vulture          ``vulture``; dead code under ``src`` at 70% confidence.
pytest           The three suites above.
sphinx           ``sphinx -W -n -E`` over ``docs``, twice: the public copy and the
                 private-members copy (see :doc:`dev_guide_api`).
================ =====================================================================

pip-audit and PyMarkdown exist in the script but are off by default (``ENABLE_PIP_AUDIT``
and ``ENABLE_PYMARKDOWN``): the first reports findings against pinned upstream
dependencies this repository does not control, and the second reports pre-existing
findings in the Markdown. ``ruff format`` is never run, because the column-aligned house
style would not survive it.

Building the documentation
--------------------------

::

    ./scripts/run-all-checks.sh --sphinx

builds both copies of this documentation into ``docs/_build/html`` and
``docs/_build/private/html``. The direct commands are::

    python -m sphinx -W -n -E -b html docs docs/_build/html
    python -m sphinx -W -n -E -t private -b html docs docs/_build/private/html

``-W`` turns every warning into an error and ``-n`` reports every cross-reference that
resolves to nothing, so a docstring that names a type with no target fails the build.
The handful of names with nothing to link to are listed in ``nitpick_ignore`` in
``docs/conf.py``, each with the reason; never add a name this package owns there. Give it
an API reference entry instead.

Continuous integration
----------------------

Two workflows run on every push and pull request:

``run-lint.yml``
    The gates above minus the test suites, each as its own step so that a failure names
    itself, on a self-hosted Linux runner under Python 3.12. It ends with the same two
    Sphinx builds.

``run-tests.yml``
    The three test suites, through ``scripts/automated_tests/oops_main_test.sh``, on the
    self-hosted runner under Python 3.11, 3.12 and 3.13, with coverage uploaded from the
    3.13 job. The script reinstalls the dependencies and needs ``SPICE_PATH``,
    ``SPICE_SQLITE_DB_NAME`` and ``OOPS_RESOURCES`` from the runner.

``run-windows-tests.yml`` covers Windows. When the set of gates changes, change the check
script first and bring the workflow into step in the same change.

Releases
--------

Versions come from ``setuptools_scm``; never hand-edit ``src/oops/_version.py``. A release
is a tag ``v<MAJOR>.<MINOR>.<PATCH>`` on ``main``. Creating a GitHub Release from that tag
runs ``publish_to_pypi.yml``, which builds and uploads the wheel;
``publish_to_test_pypi.yml`` does the same against Test PyPI.

Contributing a change
---------------------

Every change reaches ``main`` through a pull request that passes CI. Work on a branch
named ``<initials>_<YYMMDD>_<topic>`` off ``main``, with no ``feature/`` or ``bugfix/``
prefix and no release or develop branches. Commit subjects are capitalized imperative
sentences under 72 characters, with no type prefix and no trailing period; one logical
change per commit. Pull requests follow ``.github/pull_request_template.md`` (purpose,
changes, testing, potential impacts, checklist) and are squash-merged, so the subject of
the squashed commit is the pull request title. ``CONTRIBUTING.md`` covers issues and the
code of conduct.
