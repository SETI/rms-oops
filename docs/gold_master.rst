``programs.gold_master`` package
================================

The gold master backplane test framework. It computes the backplanes of a standard
observation and compares them, array by array, against a stored set of reference
arrays — the *gold masters*. It is a runnable tool rather than part of the ``oops`` API,
so it lives outside ``src`` and nothing under ``src`` imports it; that is what keeps the
test scaffolding out of the published wheel.

Two entry points drive the same comparison:
:func:`~programs.gold_master.execute_as_pytest`, which a host test calls to check one
observation, and :func:`~programs.gold_master.execute_as_command`, which exposes the
comparison as a command-line program.

Resources and environment
-------------------------

Nothing runs without the external resource tree. ``$OOPS_RESOURCES`` points at the
directory holding the SPICE kernels, the test data and the gold masters, and the
remaining variables default to subdirectories of it:

=============================== ==============================================
Variable                        Default
=============================== ==============================================
``OOPS_RESOURCES``              (required)
``OOPS_TEST_DATA_PATH``         ``$OOPS_RESOURCES/test_data``
``OOPS_GOLD_MASTER_PATH``       ``$OOPS_RESOURCES/gold_master``
``OOPS_BACKPLANE_OUTPUT_PATH``  the current directory
=============================== ==============================================

:mod:`programs.gold_master.test_support` resolves each of these into a ``filecache``
prefix, so any of them may name a cloud resource such as
``gs://rms-oops-resources/gold_master`` rather than a local directory.

Directory layout
----------------

The gold master tree and the output tree share one layout, so a directory of generated
backplanes can serve as the masters of a later run. The files for one observation live
in::

    <path>/<mission>.<instrument>/<basename>

for example ``<path>/cassini.iss/W1573721822_1``. The directory is named for the mission
and the instrument alone, not for the module's place in any import tree, so the files
stay put when the module moves.

Running under pytest
--------------------

A host test registers its standard observation and then calls
:func:`~programs.gold_master.execute_as_pytest`. The whole host suite runs with::

    pytest tests/hosts

Pass ``--gold-master`` to compare against masters somewhere other than the configured
tree; it overrides ``$OOPS_GOLD_MASTER_PATH`` and ``$OOPS_RESOURCES`` for that run
alone::

    pytest tests/hosts --gold-master=/path/to/masters

Running from the command line
-----------------------------

Each instrument also has a runnable module that calls
:func:`~programs.gold_master.execute_as_command`::

    export PYTHONPATH=.
    python tests/hosts/cassini/iss/gold_master.py --help

Three tasks select what the run does: ``--compare`` (the default) checks the computed
backplanes against the masters, ``--preview`` computes and writes them without comparing,
and ``--adopt`` writes a full set of masters, including the ``summary.py`` that holds the
backplanes whose value is constant. Naming a directory with ``--gold-master`` is what
keeps ``--adopt`` away from the real masters, which it would otherwise overwrite in
place::

    python tests/hosts/cassini/iss/gold_master.py --adopt --gold-master=/path/to/new
    python tests/hosts/cassini/iss/gold_master.py --gold-master=/path/to/new

Other options select the targets (``--planet``, ``--moon``, ``--ring``), tune the
comparison (``--tolerance``, ``--radius``, ``--ignore-missing``, ``--suite``), control
what is written (``--arrays``, ``--browse``, ``--undersample``, ``--output``) and set the
logging (``--verbose``, ``--log``, ``--level``, ``--diagnostics``). ``--help`` lists them
all with their defaults.

Test suites
-----------

Each backplane family is registered as a named test suite through
:func:`~programs.gold_master.register_test_suite`, and importing
``programs.gold_master.all`` registers the full set: ``ansa``, ``border``, ``distance``,
``lighting``, ``limb``, ``orbit``, ``pole``, ``resolution``, ``ring``, ``sky``,
``spheroid`` and ``where``. Use ``--suite`` to run a subset.

API reference
-------------

.. automodule:: programs.gold_master
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__

``programs.gold_master.test_support``
-------------------------------------

Resolves every ``$OOPS_*`` resource path into a ``filecache`` prefix, for both the tool
and the tests.

.. automodule:: programs.gold_master.test_support
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__
