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
label, it parses the label as it is in the file, trying the "fast" parser, then "strict",
then "loose". A parser that reports a syntax error, or a value it cannot convert, such as
a time with an hour of 25, is followed by the next. A label that none of them can read is
offered to each host's correction of label text, described below, and goes to the first
host whose correction makes it readable and that recognizes it. If there is no such
host, the file goes to the host that recognizes it by its path. If no host does, the
error gives the path of the label and each parser's message. A file that has already
been parsed, as a PDS3 label, a VICAR label or image, or a FITS HDU list, is identified
from that content and passed to the host as it is. Any other argument raises
:class:`~oops.OopsTypeError`.

Correcting label text
---------------------

Some labels in the archives hold text that the PDS3 parser cannot read, such as a line
of plain prose outside any comment, or that it reads wrongly. The oops readers leave the
repairs that one data set needs to its host, which overrides the class method
``_correct_label_text(text, filepath)``, which receives the text of a label, up to its
END statement for an attached label, and the path of its file, and returns the corrected
text. The default returns the text unchanged. The same method serves the product labels
that the host reads and the labels of its index and supplement, and the path tells them
apart where that matters.

Every oops method that parses a label for the host calls it before any parser runs. The
corrected text is parsed in place of the file's. The parsed label keeps the path of the
file, so that its pointers are found beside it, and an index table is still read from its
own file. For a host that keeps the default, the labels are parsed from their files.

The correction also sees the labels of other hosts. When a file or an index is being
identified and no parser can read its label, the label is offered to every host's
correction in turn. A correction should therefore change only the text that has the
defect it repairs, return any other text as it is, and not raise an exception. If one
does raise, that host is passed over, and the next host's correction is tried.

The subpackage also provides :func:`~oops.host.read_pds3_image_array` and
:func:`~oops.host.get_fits_image_hdu`, which hosts use to read image arrays;
:func:`~oops.host.is_pds3_null` and :data:`~oops.host.PDS3_NULL_TEXTS`, which tell a
placeholder such as "UNK" or "N/A" from a value; and the
:class:`~oops.host.HostError` exceptions that report a file a host cannot read.

Reading an index
----------------

:meth:`~oops.host.Host.from_index` reads a PDS3 index table and returns an iterator over
its observations, one for each row. The call itself reads only the labels of the index
and of its supplement, so a mistake in the arguments or in a label is reported at once.
It identifies the host from the index label, which is parsed with one parser, "fast"
unless the options select another. If the label does not say which instrument the index
describes, the first row is read to find out, and only that row. An index label that the
parser cannot read is offered to each host's correction of label text, as a product label
is, and goes to the first host whose correction makes it readable and that recognizes
the corrected label.

A volume keeps its own index in a directory named "index", in either case. A host that
reads only the Ring-Moon Systems Node metadata indexes refuses a volume's own index with
:class:`~oops.host.HostIndexError` before any row is read. If no host
recognizes the label of such an index, the first row is read only for the hosts that
might read it, and if every host is of the kind that reads only the metadata indexes, the
index is refused at once.

The host's own ``from_index`` then reads the index a few rows at a time as the iteration
proceeds, and builds each row's observation when the iteration reaches it. An index of
any size is read in this way while only a few rows are held in memory. An observation
that the caller has finished with is freed, together with its camera frame, so the memory
a pass over the index uses stays the same from the first row to the last.

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
:class:`~oops.OopsValueError` at once. When the caller gives the rows themselves as a list
or another sequence, the check is made against the first row at once. When the rows are
given as an iterator, whose rows are not known until they are read, the first row is
checked when the iteration reads it.
A row that lacks a filtered column is left out.

Errors while reading an index
-----------------------------

These are raised by the call itself:

- :class:`~oops.host.PdsHostError` for a label that cannot be parsed, and
  :class:`~oops.OopsValueError` for an invalid option of the label parser or for a value
  in the index label that the parser cannot convert, such as day 366 of a year that has
  365 days.
- :class:`~oops.host.HostIndexError` for a label that does not describe a table that can
  be read, as when it lacks the object that its table pointer refers to, and for a
  supplement or file column that the host needs and that a table lacks.
- :class:`~oops.host.HostIndexError` for a volume's own index, when the host that
  recognizes it, or every host that might read it, reads only the Ring-Moon Systems Node
  metadata indexes.
- :class:`~oops.OopsValueError` for a ``filter`` column that neither table has.
- :class:`FileNotFoundError` for a label that does not exist, including a supplement that
  the host requires.

These are raised during the iteration, when it reaches the row concerned, after every
earlier row has been yielded:

- :class:`FileNotFoundError` when the iteration starts, if a table file does not exist.
- :class:`~oops.host.HostIndexError` for a table file that holds fewer rows than its
  label says, after every row that the file holds has been yielded.
- :class:`~oops.host.HostIndexError` for a row that is longer than the label's row
  length. The rows are read ten at a time, so the error is raised when the iteration
  reaches the first row of those ten, and the rows read with the long one are not
  yielded.
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

The index tables are not checked beyond this. A value that cannot be read as the number
its column describes, such as a letter in a numeric column, makes the table reader issue
a warning and give the value as text, marked as masked; the row is yielded with it, and
the host may then
raise a ``ValueError`` or ``TypeError`` when it converts the value, or a pairing error if
the value is part of a file name. A row that is shorter than the label's row length is
read as it is.

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
:class:`~oops.OopsSpiceError` whenever it is used. The ``tracker`` and ``parallel``
overrides evaluate the frame once, when the observation is built. They leave a frame of
this kind as it is, so the error is still raised only when the pointing is used. A host
checks the kernels at the time at which the observation evaluates its frame: its
midtime, or for ``tracker`` the start, midtime, or end that the override selects, moved
by any time shift. A host does not check the kernels, and
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
