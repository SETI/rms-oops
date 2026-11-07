``oops.host`` subpackage
========================

This page documents the ``oops.host`` subpackage, which reads the data files of a mission
instrument and returns the :class:`~oops.Observation` objects that describe them. The
base class :class:`~oops.host.Host`, which is also available as ``oops.Host``, is defined
by the subpackage itself. The concrete hosts that extend it, one per instrument, are not
part of ``oops``: each mission's hosts are a separate package, such as
``rms-host-cassini`` or ``rms-host-galileo``, and each host registers itself with
:class:`~oops.host.Host` when its module is imported.

:meth:`~oops.host.Host.from_file` identifies the host of a data file from its PDS3
label, VICAR header, FITS header or file name, and :meth:`~oops.host.Host.from_index`
does the same for each row of a PDS3 index. The subpackage also provides
:func:`~oops.host.read_pds3_image_array` and :func:`~oops.host.get_fits_image_hdu`,
which hosts use to read image arrays, and the :class:`~oops.host.HostError` exceptions
that report a file a host cannot read.

.. automodule:: oops.host
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__
