``oops`` package
================

.. The module docstring only; the classes below are listed explicitly so that they can be
   ordered and grouped. `:no-members:` negates the project-wide autodoc default.

.. automodule:: oops
    :no-members:

This page documents the classes of the top-level ``oops`` namespace, in alphabetical
order. The concrete subclasses of each abstract base are documented on their own pages:
:doc:`oops.cadence <oops_cadence>`, :doc:`oops.calibration <oops_calibration>`,
:doc:`oops.fov <oops_fov>`, :doc:`oops.frame <oops_frame>`,
:doc:`oops.gravity <oops_gravity>`, :doc:`oops.lightsource <oops_lightsource>`,
:doc:`oops.observation <oops_observation>`, :doc:`oops.path <oops_path>`, and
:doc:`oops.surface <oops_surface>`.

.. Keep this list alphabetical and in step with `__all__` in src/oops/__init__.py.
   `Oops` and `Mutable` are named by their defining module because they are not
   re-exported into the top-level namespace.

.. autoclass:: oops.Backplane

.. autoclass:: oops.Body

.. autoclass:: oops.Cache

.. autoclass:: oops.Cadence

.. autoclass:: oops.Calibration

.. autoclass:: oops.Event

.. autoclass:: oops.Fittable

.. autoclass:: oops.FOV

.. autoclass:: oops.Frame

.. autoclass:: oops.Gravity

.. autoclass:: oops.Meshgrid

.. autoclass:: oops.mutable.Mutable

.. autoclass:: oops.Observation

.. autoclass:: oops.oops.Oops

.. autoclass:: oops.Path

.. autoclass:: oops.Surface

.. autoclass:: oops.Transform

PolyMath classes
----------------

``oops`` re-exports these array types from the ``polymath`` package. Every geometric
quantity in the library is one of them; they are documented here because they appear
throughout the API above.

.. autoclass:: oops.Boolean

.. autoclass:: oops.Matrix

.. autoclass:: oops.Matrix3

.. autoclass:: oops.Pair

.. autoclass:: oops.Quaternion

.. autoclass:: oops.Qube

.. autoclass:: oops.Scalar

.. autoclass:: oops.Vector

.. autoclass:: oops.Vector3
