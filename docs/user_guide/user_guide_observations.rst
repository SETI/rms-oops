Observations
============

Loading an observation
----------------------

Each supported instrument has a *host* module under ``oops.hosts`` whose ``from_file``
function reads a data file and returns an :class:`~oops.Observation`. The Cassini
imaging camera is the model:

.. code-block:: python

    import oops
    from oops.hosts.cassini import iss

    obs = iss.from_file('/path/to/W1573721822_1.IMG')

The first call for a mission furnishes its SPICE kernels, which takes a few seconds; a
later call in the same session is quick. ``from_file`` accepts a string, a
``pathlib.Path`` or a ``filecache`` path, so a file in cloud storage works too. The
keywords vary by instrument and are listed in :doc:`user_guide_appendix_hosts`; the ones
that recur are ``planets``, to restrict the kernels to one planet, ``asof``, to use only
kernels available by a given date, and for cameras ``fast_distortion``, which selects
a pre-inverted distortion polynomial (``True``, the default), a dynamically solved one
(``False``) or no distortion at all (``None``).

The result for a camera is a :class:`~oops.observation.Snapshot`, a 2-D image exposed at
one time. Some hosts return a list, when a file holds several observations (JunoCam's
framelets, or a VIMS cube's visible and infrared halves), and a
:class:`~oops.observation.TimedImage` when the pixels of one image have distinct times.

What an observation holds
-------------------------

For the Cassini image above:

.. code-block:: python

    >>> type(obs).__name__
    'Snapshot'
    >>> obs.data.shape, obs.data.dtype
    ((1024, 1024), dtype('uint8'))
    >>> obs.time, obs.midtime
    ((248300548.68971696, 248300548.78971696), 248300548.73971695)
    >>> obs.path.path_id, obs.frame.frame_id
    ('CASSINI', 'CASSINI_ISS_WAC')
    >>> type(obs.fov).__name__, obs.fov.uv_shape
    ('PolynomialFOV', Pair(1024 1024))
    >>> obs.u_axis, obs.v_axis, obs.swap_uv
    (1, 0, True)

``data``
    The image array, as read from the file. Nothing in the geometry uses it; it is
    carried so that a backplane and the data it describes travel together.
``time``, ``midtime``
    The start and end of the exposure, and its midpoint, in seconds TDB. They come from
    the observation's :class:`~oops.Cadence`, ``obs.cadence``.
``fov``
    The :class:`~oops.FOV`: the mapping from pixel coordinates to lines of sight,
    including distortion. ``uv_shape`` is the size in pixels and ``uv_scale`` the
    angular size of a pixel in radians.
``path``, ``frame``
    The instrument's position and pointing, as a :class:`~oops.Path` and a
    :class:`~oops.Frame` looked up from SPICE.
``uv_shape``, ``u_axis``, ``v_axis``, ``swap_uv``
    The image size in *(u,v)* order, and which array axis is which. Here the array is
    indexed ``[v, u]``, as a VICAR image is.
``subfields``
    Everything else the host attached, each also available as an attribute: for Cassini,
    ``instrument``, ``detector``, ``filter1``, ``filter2``, ``sampling``, ``gain_mode``,
    the label dictionary ``dict``, the ``filespec`` and ``basename``, and the list of
    ``spice_kernels`` used.

Pixels, lines of sight and times
--------------------------------

Observation methods translate between array indices, *(u,v)* coordinates, lines of
sight and times. Each accepts arrays as well as single values.

.. code-block:: python

    (uv, t) = obs.uvt((511.5, 511.5))         # array indices to (u,v) and time
    (t0, t1) = obs.time_range_at_uv(uv)      # the exposure interval of a pixel
    t = obs.midtime_at_uv(uv)                # its midpoint

    los = obs.fov.los_from_uv(uv)            # the line of sight, a unit Vector3 in the
                                             # observation frame
    uv = obs.fov.uv_from_los(los)            # and back
    outside = obs.fov.uv_is_outside(uv)      # Boolean, True beyond the edge

Where a target falls in the field of view:

.. code-block:: python

    uv = obs.uv_from_path('MIMAS')                       # the (u,v) of a body's center
    uv = obs.uv_from_ra_and_dec(ra, dec)                 # of a sky position, radians
    uv = obs.uv_from_coords(surface, (radius, lon))      # of a point on a surface

And which bodies are in view at all, treating each as a sphere:

.. code-block:: python

    >>> obs.inventory(['SATURN', 'EPIMETHEUS', 'MIMAS', 'ENCELADUS', 'TITAN'])
    ['SATURN', 'EPIMETHEUS']
    >>> full = obs.inventory(['SATURN', 'EPIMETHEUS'], return_type='full')
    >>> full['EPIMETHEUS']['center_uv'], full['EPIMETHEUS']['range']
    (array([502.24, 523.67]), 1555030.4)

With ``return_type='full'`` each entry has the body's ``center_uv``, ``range`` in km,
``resolution`` in km per pixel, the bounding box ``u_min``, ``u_max``, ``v_min``,
``v_max`` clipped to the image and unclipped, and ``inside``, whether any of it is in
view. ``expand`` widens each body by a number of pixels, and ``tfrac`` or ``time``
selects the moment within the exposure. :class:`~oops.Backplane` uses the same inventory
to confine its work to the pixels that can see each body.

Meshgrids
---------

A :class:`~oops.Meshgrid` is the set of pixel coordinates at which backplanes are
evaluated, with the lines of sight cached. ``obs.meshgrid()`` samples the center of
every pixel, which is what a backplane uses by default; the options thin it out or move
it:

.. code-block:: python

    mg = obs.meshgrid()                              # every pixel center, (1024, 1024)
    mg = obs.meshgrid(undersample=8)                 # every eighth pixel, (128, 128)
    mg = obs.meshgrid(oversample=2)                  # four samples per pixel
    mg = obs.meshgrid(origin=(0.5, 0.5), limit=(512, 512))   # the upper-left quadrant
    mg = oops.Meshgrid.for_fov_center(obs.fov)       # the boresight alone, shape ()
    mg = oops.Meshgrid(obs.fov, [(100.5, 200.5), (300.5, 400.5)])   # chosen pixels

Undersampling by 16 reduces a Cassini image to 64 by 64 samples and cuts the time to
compute a backplane by two orders of magnitude, which is why the gold master tool
compares at that resolution and computes at full resolution only when asked to write
arrays.

Timing
------

For a snapshot every pixel shares the exposure, and a backplane is evaluated at
``midtime``. For a timed image, ``obs.timegrid(meshgrid)`` gives the time of every sample
of a meshgrid, and a backplane is evaluated at those; ``tfrac`` selects a point within
each pixel's exposure. ``obs.cadence`` is the underlying :class:`~oops.Cadence`, with
``time_at_tstep`` and ``tstep_at_time`` to convert between time steps and times.
``obs.time_shift(seconds)`` returns a copy of the observation shifted in time, for
testing the sensitivity of the geometry to a timing error.

Pointing corrections
--------------------

The pointing from a SPICE C-kernel can be off by a few pixels. ``obs.navigate((dx, dy))``
applies a small rotation, in radians, and returns the corrected observation, and
``obs.set_frame(frame)`` substitutes a frame of your own. The Cassini host's
``navigation=True`` wraps the frame in a fittable :class:`~oops.frame.Navigation` frame
whose parameters a fitting procedure can adjust.
