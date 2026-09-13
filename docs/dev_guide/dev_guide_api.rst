API reference for developers
============================

The API reference is generated from the docstrings by autodoc, one page per package:
the :doc:`oops package </oops>` with its abstract classes, :class:`~oops.Body`,
:class:`~oops.Event`, :class:`~oops.Backplane`, :class:`~oops.Meshgrid`,
:class:`~oops.Transform` and the PolyMath types; the subpackage pages
:doc:`oops.cadence </oops_cadence>`, :doc:`oops.calibration </oops_calibration>`,
:doc:`oops.config </oops_config>`, :doc:`oops.fov </oops_fov>`, :doc:`oops.frame
</oops_frame>`, :doc:`oops.gravity </oops_gravity>`, :doc:`oops.lightsource
</oops_lightsource>`, :doc:`oops.observation </oops_observation>`, :doc:`oops.path
</oops_path>` and :doc:`oops.surface </oops_surface>`; the :doc:`gold master framework
</gold_master>`; and :doc:`spicedb </spicedb>`.

Two copies of the reference
---------------------------

The documentation is built twice from the same tree. The published copy documents the
public API alone. The second copy, built with the Sphinx tag ``private``, documents the
private members as well: on every class page, the methods and attributes whose names
begin with an underscore, such as ``Backplane._define_backplane_names``,
``Frame._register`` or ``Event._refresh``, appear alongside the public ones, and the
sections below fill in with the private modules that no public page covers. A developer
changing the library works from that copy, because the private members are what a
subclass, a backplane method or a test touches. The check script builds both::

    ./scripts/run-all-checks.sh --sphinx

or directly::

    python -m sphinx -W -n -E -t private -b html docs docs/_build/private/html

.. only:: private

   **This is the private copy.** Every page of the API reference in this build shows the
   private members, and the sections below document the private modules.

.. only:: not private

   **This is the public copy.** The private members and the sections below are absent
   here; build the private copy as shown above to see them.

The narrative chapters of this guide name private members in plain monospace,
``Path._register`` for instance, rather than as links, because a link to a private
member would be broken in the public copy. Find them on the class's page in the private
copy.

Private modules
---------------

The modules below are private, or hold private helpers behind a public class, and so
have no place in the public reference. In the private copy they are documented here.

.. private-only::

   The photon solvers
   ~~~~~~~~~~~~~~~~~~

   The functions bound onto :class:`~oops.Path` and :class:`~oops.Surface` at import.
   ``photon_to_event`` and ``photon_from_event`` are documented as methods of those
   classes; the solver bodies and their helpers are here.

   .. automodule:: oops.path._photon_solver
       :members:
       :private-members:
       :exclude-members: photon_to_event, photon_from_event

   .. automodule:: oops.surface._photon_solver
       :members:
       :private-members:
       :exclude-members: photon_to_event, photon_from_event, photon_to_coords,
           photon_from_coords, photon_normal_to_event, photon_event_to_normal,
           photon_path_to_normal, photon_normal_to_path

   Per-ray convergence
   ~~~~~~~~~~~~~~~~~~~

   The bookkeeping behind the iterative solvers, which judge every ray on its own: the
   photon solvers, the ground-point and limb solvers of :class:`~oops.surface.Limb`,
   :class:`~oops.surface.Spheroid` and :class:`~oops.surface.Ellipsoid`, and the
   distortion solvers of :class:`~oops.fov.BarrelFOV` and
   :class:`~oops.fov.PolynomialFOV`.

   .. automodule:: oops._convergence
       :members:
       :private-members:

   The bounded cache
   ~~~~~~~~~~~~~~~~~

   .. automodule:: oops._cache
       :members:
       :private-members:

   The mutable protocol
   ~~~~~~~~~~~~~~~~~~~~

   The module-level functions behind :class:`~oops.mutable.Mutable`, which is documented
   on the :doc:`oops package page </oops>`.

   .. automodule:: oops.mutable
       :members:
       :private-members:
       :exclude-members: Mutable

   Body definition helpers
   ~~~~~~~~~~~~~~~~~~~~~~~

   The private module-level functions of ``oops/body.py`` that
   :meth:`~oops.Body.define_solar_system` calls; the class itself is on the
   :doc:`oops package page </oops>`.

   .. automodule:: oops.body
       :members:
       :private-members:
       :exclude-members: Body

   Gold master internals
   ~~~~~~~~~~~~~~~~~~~~~

   The private class that carries one comparison through the gold master framework. The
   public API is on the :doc:`gold master page </gold_master>`, whose ``automodule``
   directive excludes this class so that this is its one, canonical entry.

   .. autoclass:: programs.gold_master._BackplaneComparison
       :members:
       :private-members:

The stubs
---------

The published type information is not in the source, whose methods are typed in their
docstrings, but in one ``__init__.pyi`` per package, which declares every public class
and function outright. A name is therefore typed when it is imported from the package
(``from oops.frame import SpinFrame``) and not when it is imported from the defining
module. The stubs and the docstrings must agree, and ``stubtest`` checks the stubs
against the run-time API; the members bound at import time, the class constants
assigned after a class statement, and the 86 backplane methods are all written into the
stubs by hand. ``stubtest-allowlist.txt`` lists the names that exist only at run time,
chiefly the cross-class placeholders that ``oops/__init__.py`` fills.
