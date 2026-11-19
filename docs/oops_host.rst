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
identifies the host of a PDS3 index from its label, or else from its first row, and
passes the index to that host's own ``from_index``. ``from_index`` returns an iterator:
it reads and filters the index at once, and builds each row's observation when the
iteration reaches that row. The subpackage also provides
:func:`~oops.host.read_pds3_image_array` and :func:`~oops.host.get_fits_image_hdu`, which
hosts use to read image arrays, and the :class:`~oops.host.HostError` exceptions that
report a file a host cannot read.

Many indexes have a supplemental index, a second table with more columns for the same
products, whose columns are merged into each row. When the two tables list the same files
in the same order, each index row is merged with the supplement row in the same position.
When they list them in different orders, as many archive volumes do, each index row is
merged with the supplement row that names the same file. Paths are compared without
regard to case, to a leading "/" or "./", or to the extension, so a row that names a
product's label matches a row that names its data file. The call raises
:class:`~oops.host.HostIndexError` if an index row has no supplement row, if a supplement
row belongs to no index row, or, when the rows are paired by file, if either table lists a
file twice. The error names the first few such files and both tables.

A host that cannot obtain an observation's pointing from the SPICE kernels raises
:class:`~oops.OopsSpiceError`, which carries the SPICE error code and the file being
read. It does not check the kernels, and so does not raise that exception, when the
caller supplies the pointing, either as a ``frame`` override or by choosing a pointing
source other than SPICE. The rule is given in full under :class:`~oops.host.Host`.

.. automodule:: oops.host
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__
