``oops.host`` subpackage
========================

This page documents the ``oops.host`` subpackage, which reads the data files of a mission
instrument and returns the :class:`~oops.Observation` objects that describe them. The
base class :class:`~oops.host.Host`, which is also available as ``oops.Host``, is defined
by the subpackage itself. The concrete hosts that extend it, one per instrument, are not
part of ``oops``: each mission's hosts are a separate package, such as
``rms-host-cassini`` or ``rms-host-galileo``, and each host registers itself with
:class:`~oops.host.Host` when its module is imported.

Reading a file
--------------

:meth:`~oops.host.Host.from_file` identifies the host of a data file from its PDS3
label, VICAR header, FITS header, or file name, and passes the file's path, with every
other argument as given, to that host's own ``from_file``, which reads the file itself.
The result is therefore the same as calling the host class directly. To identify a PDS3
label, it tries the "fast" parser, then "strict", then "loose". A label that none of them
can read goes to the host that recognizes it by its path, if there is one. A file that
has already been parsed, as a PDS3 label, a VICAR label or image, or a FITS HDU list, is
identified from that content and passed to the host as it is. Any other argument raises
:class:`~oops.OopsTypeError`.

The subpackage also provides :func:`~oops.host.read_pds3_image_array` and
:func:`~oops.host.get_fits_image_hdu`, which hosts use to read image arrays, and the
:class:`~oops.host.HostError` exceptions that report a file a host cannot read.

Reading an index
----------------

:meth:`~oops.host.Host.from_index` reads a PDS3 index table and returns an iterator over
its observations, one for each row. The call itself reads only the labels of the index
and of its supplement, so a mistake in the arguments or in a label is reported at once.
It identifies the host from the index label. If the label does not say which instrument
the index describes, the first row is read to find out, and only that row. The host's own
``from_index`` then reads the index a few rows at a time as the iteration proceeds, and
builds each row's observation when the iteration reaches it. An index of any size is read
in this way while only a few rows are held in memory. An observation that the caller has
finished with is freed, together with its camera frame, so the memory a pass over the
index uses stays the same from the first row to the last.

Many indexes have a supplemental index, a second table with more columns for the same
products, whose columns are merged into each row. How the two tables are paired is fixed
for each host, from a survey of that instrument's indexes:

- If every index of the instrument lists its products in the same order as its
  supplement, each index row is paired with the supplement row in the same position. The
  supplement is read in step with the index. If the two rows describe different files,
  the iteration stops at that row with :class:`~oops.host.HostIndexError`.
- If some supplements list the products in another order, each index row is paired with
  the supplement row for the same file. The supplement is read ahead only as far as that
  row, and a supplement row that is read early is held until its index row is reached.
  Because the rows are seldom far out of order, few rows are held at any time.

Paths are compared without regard to case, to a leading "/" or "./", or to the extension,
so a row that gives a product's label matches a row that gives its data file. Each pair of
rows is also checked, by default by comparing the columns that both tables have.

A ``filter`` selects rows by the values of their columns, and each row is tested as it is
read. A ``filter`` that uses a column that neither table has raises
:class:`~oops.OopsValueError` at once. A row that lacks a filtered column is left out.

Errors while reading an index
-----------------------------

These are raised by the call itself:

- :class:`~oops.host.PdsHostError` for a label that cannot be parsed, and
  :class:`~oops.OopsValueError` for an invalid option of the label parser.
- :class:`~oops.host.HostIndexError` for a label that does not describe a table that can
  be read, as when it lacks the object that its table pointer refers to, and for a
  supplement or file column that the host needs and that a table lacks.
- :class:`~oops.host.HostIndexError` for a volume's own index, from a host that reads
  only the Ring-Moon Systems Node metadata indexes.
- :class:`~oops.OopsValueError` for a ``filter`` column that neither table has.
- :class:`FileNotFoundError` for a label that does not exist, including a supplement that
  the host requires.

These are raised during the iteration, when it reaches the row concerned, after every
earlier row has been yielded:

- :class:`FileNotFoundError` when the iteration starts, if a table file does not exist.
- :class:`~oops.host.HostIndexError` for a row that the table file does not hold or that
  does not match the label's description of the table. The rows are read ten at a time,
  so a row that does not match the label also holds back the rows read with it.
- :class:`~oops.host.HostIndexError` for a pair of rows that describe different files, a
  pair that fails the host's check, or an index row for which the supplement lacks a row.
  When the rows are paired by file, a missing supplement row is found only after the
  whole supplement has been read. A file that the index lists more times than the
  supplement is reported in the same way.
- :class:`~oops.host.HostIndexError` after the last index row, for a supplement row that
  was not paired with any index row, or a table file that holds more rows than its label
  says. A file that the supplement lists more times than the index is reported as a
  supplement row that was not paired, unless the second copy is read while the first is
  still held, which is reported at once.
- :class:`~oops.OopsTypeError` for a row whose value cannot be compared with a range in
  ``filter``.
- Any error that the host raises when it builds an observation from a row.

The iteration ends at the first error.

File subfields
--------------

Each observation gives its file in four subfields: ``filepath``, the path as given;
``basename``, the file name; ``image_url``, the absolute path or URL; and ``abspath``, the
path of the file on the local disk. ``abspath`` is present only when the file is there:
a local file that exists, or a remote file that is already in the local cache. Building
an observation never creates a directory, so reading an index whose data files are not
on the local disk leaves the disk unchanged.

Pointing from SPICE
-------------------

A host that cannot obtain an observation's pointing from the SPICE kernels raises
:class:`~oops.OopsSpiceError`, which carries the SPICE error code and the file being
read. ``from_file`` raises it when it builds the observation. In ``from_index``, every
row gives an observation, and a row whose pointing SPICE cannot supply gets a
:class:`~oops.frame.SpiceErrorFrame` in place of its pointing. That frame raises
:class:`~oops.OopsSpiceError` whenever it is used. A host does not check the kernels, and
so does not raise that exception, when the caller supplies the pointing, either as a
``frame`` override or by choosing a pointing source other than SPICE. The rule is given in
full under :class:`~oops.host.Host`.

API reference
-------------

.. automodule:: oops.host
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__
