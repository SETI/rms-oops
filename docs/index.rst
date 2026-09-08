Documentation for rms-oops
==========================

``rms-oops`` is an observation-geometry library for planetary science. It models
instruments, spacecraft trajectories, and target bodies, and computes per-pixel geometry
("backplanes") for real observations. It is built on SPICE, through ``cspyce``, and on
the ``polymath`` array types.

The library itself is the :mod:`oops` package, with :mod:`spicedb` alongside it to
resolve SPICE kernels. :mod:`programs.gold_master` is a separate, runnable tool that
regression-tests the backplanes of a standard observation against stored reference
arrays; it is documented here because host tests call into it, but it is not part of the
``oops`` API and does not ship in the wheel.

The :doc:`User's Guide <user_guide/user_guide>` shows how to load an observation,
describe the solar system around it and compute its geometry. The :doc:`Developer's Guide
<dev_guide/dev_guide>` explains how the library is put together and how to extend and
test it.

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   user_guide/user_guide
   dev_guide/dev_guide

.. toctree::
   :maxdepth: 2
   :caption: API reference:

   oops
   oops_cadence
   oops_calibration
   oops_config
   oops_fov
   oops_frame
   oops_gravity
   oops_lightsource
   oops_observation
   oops_path
   oops_surface
   gold_master
   spicedb

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
