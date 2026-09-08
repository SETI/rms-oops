Coding conventions
==================

These are the rules the linters cannot fully enforce. ``CLAUDE.md`` at the repository
root is the working reference for them and for the architecture traps; this chapter
restates what a new contributor needs before the first pull request.

Layout of a file
----------------

* Lines are at most 90 columns, everywhere. The one exception is a bare URL in a host
  module that cannot be wrapped.
* Every file opens with a banner of exactly 90 ``#`` characters, then a line naming the
  path from the import root (``# oops/frame/spinframe.py``, never ``src/oops/...``;
  ``# programs/gold_master/pole.py``), optionally followed by a colon and a description,
  then the banner again. The last line of the file is a closing banner of the same width.
  Every horizontal rule made of ``#`` is 90 columns wide, indent included. A lone ``#``
  on a line of its own is a blank line inside a comment paragraph, not a rule.
* There are no ``#===...`` or ``#---...`` separator rules above functions. They used to
  sit above each ``def`` in the legacy modules; do not reintroduce them.
* Imports are absolute. ``polymath`` is grouped with the ``oops`` imports, not with the
  third-party ones, and the ``import`` keywords of a group are aligned in a column.
* Column-aligned assignments, imports and trailing comments are the house style.
  ``ruff format`` is never run, and alignment or blank-line counts that the linters
  accept are not to be "fixed".

Names
-----

* A subpackage's defining module carries a trailing underscore (``frame_.py``); subclass
  modules are lowercase with no separators (``twovectorframe.py``).
* Private names take a single leading underscore: ``_register``, ``_FRAME_REGISTRY``,
  ``_photon_solver.py``. The registries and their registration hooks are private.
* Every ``__init__.py`` declares ``__all__``. A module that exists for its import side
  effects alone, such as ``backplane/all.py``, carries ``# flake8: noqa: F401`` instead.

Docstrings and types
--------------------

Two docstring styles coexist. The modern style is Google-like: ``Parameters:`` (never
``Args:``), ``Returns:``, ``Raises:``, a class-level ``Attributes:`` block, and a
noun-phrase summary line. The legacy style has two-column ``Input:`` and ``Return:``
blocks hanging at column 25. Match the file you are in.

Under ``src``, ordinary methods and functions carry no signature annotations. Every
parameter and return is typed in the docstring instead: ``name (type): ...`` under
``Parameters:`` and ``type: ...`` under ``Returns:``, with ``A | B`` for alternatives
(never ``A or B``), ``None`` last, and containers parameterized where the content is
known. A parameter that accepts a PolyMath class, a number, or a nested sequence, because
the body passes it through ``as_scalar`` or a sibling, is documented with the matching
alias from ``polymath.typedefs`` (``ScalarLike``, ``PairLike``, ``Vector3Like``); a return
is documented as the exact class. The only inline annotations are on properties: every
getter is annotated ``-> T`` and every setter ``-> None``, a name not yet bound when the
class body runs is quoted, and a class that cannot be imported without a cycle is
imported under ``TYPE_CHECKING``.

The stubs repeat the docstring types, and the two must agree: a parameter documented as
``ScalarLike`` is annotated ``ScalarLike`` in ``__init__.pyi``, and changing one means
changing the other. ``Any`` remains only where the docstring gives no type.

A method, function, class or attribute named in docstring prose takes a Sphinx role
(``:meth:``, ``:func:``, ``:class:``, ``:attr:``) so that the API reference links it; bare
single backticks are for parameter names, which the build renders in bold to match the
``Parameters:`` block. Double backticks mark code expressions. Docstrings describe the
current behavior only: no change history, no ticket numbers, no "new" or "legacy".

Every class documents its constructor arguments in the ``__init__`` docstring, not in
the class docstring; the build appends the one to the other.

Tests
-----

Tests use pytest: module-level ``test_*`` functions, plain ``assert``, fixtures rather
than setup methods, and full type annotations including ``-> None``, because mypy checks
``tests``. There are no ``unittest.TestCase`` classes and no ``runTest`` methods. Shared
fixtures live in ``tests/conftest.py``; setup specific to one module stays in that module
as an ``autouse`` fixture. :doc:`dev_guide_testing` has the details.

Traps
-----

* Always ``import oops`` first. The bottom of ``oops/__init__.py`` injects the
  cross-class attributes that break the circular imports; importing a leaf module on its
  own leaves them ``None`` and fails obscurely far from the cause.
* Geometry values are PolyMath types, not NumPy arrays, and the data under a mask is
  garbage: combine with ``.antimask`` before using ``.vals``.
* Units are bare km, seconds TDB and radians. ``polymath.Units`` is effectively unused,
  so a returned value is not self-describing, and angles are radians even where a label
  says "deg".
* :class:`~oops.Event` objects and registered backplane arrays are read-only and shared
  through caches. Mutating one in place corrupts every other user of it.
* ``quick=True`` disables the ``QuickPath`` and ``QuickFrame`` optimization; ``quick``
  accepts ``None``, a dictionary of overrides, or ``False``.
* A backplane module's every module-level function is swept into
  :class:`~oops.Backplane`; do not leave stray helpers there.
