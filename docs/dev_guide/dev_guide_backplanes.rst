The backplane engine
====================

Overview
--------

:class:`~oops.Backplane` computes the geometry of every pixel of an observation. It owns
the observation and a :class:`~oops.Meshgrid`, builds the arrival event of every sample,
solves the photon path back to each surface a caller asks about, and caches every
intermediate event and every resulting array under a key, so that the second backplane
about a surface costs a fraction of the first. The class itself, in
``oops/backplane/__init__.py``, holds the keys, the event retrieval and the caches; the
backplane methods live in one module per family and are attached to the class when
``oops.backplane.all`` is imported. Its API reference is on the :doc:`oops package page
</oops>`.

Keys
----

Everything is addressed by a key, and there are two kinds.

An *event key* names a photon event at a surface: a tuple ``(source_key, surface_key)``,
or ``(source_key, surface_key, shadowing_surface_key)`` for a shadow. The surface key is
a body name, optionally suffixed ``:RING``, ``:ANSA`` or ``:LIMB`` to select the ring
plane, ansa or limb surface associated with the body rather than its own surface; a
registered ring body such as ``SATURN_MAIN_RINGS`` is a surface key in its own right. The
source key is the name of a body or light source with a suffix: ``<`` for dispersed
illumination (a photon from the source scattering off the surface toward the observer,
the ordinary case), ``>`` for an occultation (the source seen through the surface), or
``-`` for a *gridless* event, one at the center of the body rather than at each pixel. A
bare string is shorthand for ``("SUN<", string)``, and the empty tuple is the observation
itself. ``standardize_event_key`` normalizes any of these, upper-casing and validating;
``gridless_event_key`` rewrites the source suffix to ``-``; the private predicates
``_is_dispersed``, ``_is_occultation``, ``_is_gridless`` and ``_is_shadowing`` classify a
standardized key.

A *backplane key* names an array: ``(method_name, event_key, *extra_args)``, with the
extra arguments in the order the method takes them, so
``('ring_longitude', ('SUN<', 'SATURN:RING'), 'obs')`` is the longitude measured from the
observer. ``evaluate`` dispatches such a key to the method, after checking that the name
is in ``Backplane._CALLABLES``, and ``get_backplane`` returns a registered array or
raises ``KeyError``. A registered array carries its own key as the attribute ``key``, so
an array can be passed wherever a backplane key is accepted;
``standardize_backplane_key`` handles all three forms.

The event caches
----------------

``get_obs_event(event_key)`` is the arrival event at the instrument for the meshgrid, or
the gridless event for a ``-`` key; ``get_surface_event(event_key, derivs=False,
arrivals=False)`` is the event at the surface, solved from the observation event by the
surface's photon solver, with the arriving photon from the source filled in when
``arrivals`` is true; ``get_gridless_event`` is the same for the body center. Each caches
its result in a dictionary keyed by the event key and by whether derivatives were
requested, and returns the cached event, read-only and shared, on every later call.

Two optimizations sit underneath. ``get_surface(surface_key)``, a static method with an
LRU cache, turns a surface key into the :class:`~oops.Surface` object from the body
registry. ``get_antimask(surface_key)`` returns the bounding box of the pixels that can
intercept a body, from the observation's inventory when the backplane was built with
``inventory=True``, and ``True`` otherwise; the photon solver is then run only inside
the box. A surface key with a colon never has an antimask, because the ring, ansa and
limb surfaces extend beyond the body's disk. The private ``_intercept_dict_key`` reduces
an event key to the surface's ``intercept_key``, so that surfaces differing only in a
mask or a coordinate convention share one photon solution, and ``_save_event`` files a
new event under every key that resolves to it, after inserting the ``body``,
``surface`` and ``event_key`` subfields.

Registration
------------

``register_backplane(key, array, expand=False, derivs=False)`` is how a method files its
result. It coerces a bool or a NumPy array into a :class:`~oops.Boolean` or
:class:`~oops.Scalar`, collapses a mask that is uniformly false, broadcasts a shapeless
value to the meshgrid shape when ``expand`` is true, attaches the key as the array's
``key`` attribute, makes the array read-only, and stores the version without derivatives
in ``_backplanes`` and the version with them, if any, in ``_backplanes_with_derivs``. It
returns the array with derivatives when ``derivs`` or the class flag ``_ALL_DERIVS`` is
set and without them otherwise. The gold master tool sets ``_ALL_DERIVS`` so that it can
test the spatial derivatives of every backplane against finite differences.

The module pattern
------------------

A backplane module is a file of module-level functions whose first parameter is
``self``. The last statement of the module hands the module's namespace to the class:

.. code-block:: python

    Backplane._define_backplane_names(globals().copy())

which sets every function in that namespace as an attribute of :class:`~oops.Backplane`
and records every name that does not begin with an underscore in ``_CALLABLES``. The
sweep is indiscriminate: an imported function would become a method too. So a backplane
module imports classes and constants only, and every helper it defines is either a
private function meant to be a method (``_fill_ring_intercepts``, say, which becomes
``Backplane._fill_ring_intercepts``) or does not belong there.

A method follows one shape, here ``light_time`` from ``distance.py``:

.. code-block:: python

    def light_time(self, event_key, direction='dep'):
        """Time in seconds between a photon's departure and its arrival.
        ...
        """

        if direction not in ('dep', 'arr'):
            raise ValueError('invalid photon direction: ' + repr(direction))

        self.refresh()
        event_key = Backplane.standardize_event_key(event_key)
        key = ('light_time', event_key, direction)
        if key in self._backplanes:
            return self.get_backplane(key)

        if direction == 'arr':
            event = self.get_surface_event(event_key, arrivals=True)
            lt = event.arr_lt
        else:
            event = self.get_surface_event(event_key)
            lt = event.dep_lt

        return self.register_backplane(key, lt.abs())

Validate the arguments, ``refresh``, standardize the key, build the backplane key with
the arguments in signature order, return the cached array if there is one, compute from
the events, register. A gridless variant is the same method called with the gridless
key, as ``center_light_time`` does:

.. code-block:: python

    def center_light_time(self, event_key, direction='dep'):
        gridless_key = Backplane.gridless_event_key(event_key)
        return self.light_time(gridless_key, direction=direction)

A method that takes a backplane key rather than an event key, such as the ``where_*``
and ``border_*`` families, standardizes it with ``standardize_backplane_key`` and
obtains the array through ``evaluate``.

The modules are ``ansa``, ``border``, ``distance``, ``lighting``, ``limb``, ``orbit``,
``pixel``, ``pole``, ``resolution``, ``ring``, ``sky``, ``spheroid`` and ``where``, and
``oops/backplane/all.py`` imports each of them for its side effect. A new module must be
added there, in alphabetical order, or nothing will attach its methods.
:doc:`dev_guide_extending` gives the full recipe, and the :doc:`User's Guide
</user_guide/user_guide_backplanes>` catalogs the methods by family.

Derivatives
-----------

The meshgrid carries *d(u,v)/d(u,v)*, the lines of sight carry *dlos/d(u,v)*, and the
photon solver propagates a ``los`` derivative through the intercept when ``derivs`` is
true, so that a backplane can report its own gradient across the image. ``dlos_duv``
on the backplane is the chain's first link; a gold master derivative test divides a
backplane's ``d_dlos`` through it and compares with a central difference from two
offset meshgrids. A backplane method should compute with whatever derivatives its
inputs carry and let ``register_backplane`` strip them from the shared copy.

Diagnostics
-----------

Three class flags, all off by default, are set by the gold master tool from its
command line: ``_DIAGNOSTICS`` logs each reuse of a cached intercept and the antimask
decisions, ``_PERFORMANCE`` logs the wall time of each photon solve, and
``CONVERGENCE`` turns on the iteration messages of :class:`~oops.config.LOGGING`. The
tool's ``--internals`` option dumps every cache at the end of a run.
