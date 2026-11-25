``oops.host`` subpackage
========================

This page documents the ``oops.host`` subpackage, which reads the data files of a mission
instrument and returns the :class:`~oops.Observation` objects that describe them. The
base class :class:`~oops.host.Host`, which is also available as ``oops.Host``, is defined
by the subpackage itself. The concrete hosts that extend it, one per instrument, are not
part of ``oops``: each mission's hosts are a separate package, such as
``rms-host-cassini`` or ``rms-host-galileo``, and each host registers itself with
:class:`~oops.host.Host` when its module is imported.

The first two sections describe what a caller sees. The rest describe the support
methods that a host overrides or calls. Their names begin with an underscore, because
they are not meant for callers, and their docstrings appear only in the private copy of
the API reference, so this page describes each of them in full.

Reading a file
--------------

:meth:`~oops.host.Host.from_file` asks every registered host whether it recognizes a data
file, from its PDS3 label, VICAR header, FITS header, or file name. Exactly one host must
claim it. A file that two or more hosts claim raises :class:`~oops.host.HostError`, whose
message names every one of them, so that two packages that both recognize a file are
found out, and neither is chosen by the order in which they were imported.

A PDS3 label is parsed for this as it is written, with the "strict" method of the parser
unless the ``pds3_options`` argument selects another. If the label cannot be parsed,
whether for invalid syntax or for a value that the parser cannot convert, it is offered
to the correction of label text of each registered host that has one, as described under
`Correcting label text`_. A host claims the label when its correction makes the label
readable and it then recognizes the corrected label, and two hosts that both claim it
raise :class:`~oops.host.HostError` as above. If no correction helps, the parser's first
error is raised: :class:`~oops.host.PdsHostError`, with the path of the label, for
invalid syntax, or :class:`~oops.OopsValueError` for a value that cannot be converted.

The host that claims the file then reads it with its own ``from_file``, which receives
every argument as the caller gave it, ``pds3_options`` included, so the result is the
one that calling the host class directly gives. A PDS3 label or a FITS file is handed to
the host as its path, so that the host reads it with its own method and, for a label,
its own correction; a FITS file is opened only to read its header, and it is closed again
before the host is given the path. A VICAR file is handed over as the content that was
read to identify it, and a VICAR file that is truncated or cannot be read raises
:class:`~oops.host.VicarHostError`, with a message that names the file. The ``calibrations`` argument defaults to None, as it does for
:meth:`~oops.host.Host.from_index`, so the calibrations that the host attaches are kept;
an empty list removes them.

Reading an index
----------------

:meth:`~oops.host.Host.from_index` reads a PDS3 index table and returns an iterator over
its observations, one for each row, in index order. A caller that needs ``len()`` or
indexing wraps the call in ``list()``:

.. code-block:: python

    observations = list(Host.from_index('/path/to/COISS_2001_index.lbl'))

The call parses the index label as it is written, with the "strict" method unless
``pds3_options`` selects another, and asks every registered host whether it recognizes
the label. A host that cannot tell from the label alone says so, and if no host claims the
label, the first row of the index is read, and only that row, and offered to each of
those hosts. Exactly one host must claim the index. Two or more raise
:class:`~oops.host.HostError`, naming each of them, and none raises
:class:`~oops.host.UnknownHostError`.

The call then hands the path and every argument, as the caller gave them, to the host's
own ``from_index``. A label that the caller passed is handed on, but the label parsed to
identify the host is not. The host reads the index and its supplement itself, 100 rows at
a time as the iteration proceeds, and builds each row's observation when the iteration
reaches that row. An index of any size is read in this way while only a few rows are held
in memory. An error in a row is raised when the iteration reaches that row, and the
iteration ends there. The one exception is a row that is too long: rows are read 100 at
a time, so that error is raised at the first row of the chunk that holds it, and up to 99
earlier rows of that chunk are never yielded.

Many indexes have a supplemental index: a second table with more columns for the same
products, whose columns are merged into each row. The ``supplement`` argument names it.
By default, it is the file beside the index that the host names, which for most hosts is
the file whose name matches the index's except that "_index" is replaced by
"_supplemental_index", in either case. ``supplement=None`` reads the index alone, and an
explicit path is read from whatever directory it names.

Each row is a dictionary of every column's value exactly as the table gives it,
declared null values included, together with the "_mask" entry of each column, which is
True where the value is null, could not be read as its column's type, or lies outside
the column's VALID_MINIMUM and VALID_MAXIMUM. Where the index and its supplement both
have a column, the row has the index's value. An observation built from a row names its
file by the relative path that the index gives, as described under `File subfields`_.

Writing a host
--------------

A host is a subclass of :class:`~oops.host.Host` that calls its class method
``_register()`` once it is defined. It overrides the detectors that recognize its files,
``_detect_in_pds3``, ``_detect_in_vicar``, ``_detect_in_fits``, ``_detect_in_file``,
``_detect_in_index``, and ``_detect_in_row``, each a static method that returns True for a
file of this host. ``_detect_in_index`` returns None if the index label cannot tell, in
which case ``_detect_in_row`` is asked about the first row.

A host's ``from_file`` reads its file through ``_read_fileinfo``, builds the constructor
arguments of its observation, applies the caller's overrides to them with
``_apply_overrides``, builds the observation, and names its files with
``_set_path_attributes``. A host's ``from_index`` reads its rows through ``_index_rows``
and builds one observation per row as the iteration reaches it. The sections below
describe each of these.

``_read_fileinfo(fileinfo, *, formats, astrometry=False, pds3_options=None)``
    A class method that reads and parses a file for this host and returns its parsed
    content: a :class:`~pdsparser.Pds3Label` for a PDS3 label, a
    :class:`~vicar.VicarLabel` (with ``astrometry=True``) or :class:`~vicar.VicarImage`
    for a VICAR file, an :class:`~astropy.io.fits.HDUList` for a FITS file, or the path
    itself for a file in another format. ``formats`` lists the formats to consider, as
    letters: "P" for PDS3, "V" for VICAR, "F" for FITS, and "O" for other. Content that is
    already parsed is returned as it is. A file that this host does not recognize raises
    :class:`~oops.host.WrongHostError`. A PDS3 label is parsed with ``pds3_options``
    after this host's correction of label text.

``_apply_overrides(params, *, timeshift=None, navigation=None, tracker=None, parallel=None, **kwargs)``
    A static method that applies the standard overrides of ``from_file`` and
    ``from_index`` to the dictionary of the observation's constructor arguments, which
    it changes in place and returns. ``target``, ``lightsource``, ``path``, ``frame``,
    ``cadence``, ``fov``, and ``calibrations`` replace the host's values when they are
    not None, and a ``frame`` override, such as a :class:`~oops.frame.Cmatrix`, is used
    as it is given. ``timeshift``, ``tracker``, ``parallel``, and ``navigation`` wrap the
    cadence, path, and frame as the docstring of :meth:`~oops.host.Host.from_file`
    describes. Any other keyword is ignored. The subfields that name the files are not
    set here.

File subfields
--------------

A host calls ``_set_path_attributes(obs, image_path, label_path)``, a static method, for
each observation it builds, after it has read its files. It sets the subfields that name
the observation's files.

For an observation from a file, ``image_path`` is the path of the image file, whichever
file the caller passed, and ``label_path`` is the path of the detached PDS3 label if the
host read it, and None otherwise. The subfields are:

- ``filepath``: the path of the image file, as an FCPath.
- ``basename``: the name of the image file.
- ``image_url``: the absolute path or URL of the image file.
- ``image_abspath``: the path of the image file on the local disk, if the host read the
  image file. It is set when ``label_path`` is None, because the host then read the image
  file itself; when the observation has a ``header_dict`` subfield, which is present
  exactly when the host read the label of the image file itself, and reading the header
  alone counts as reading the file; and when the observation has a ``data`` subfield,
  because the host then read its array. An observation built from a detached label
  alone, for astrometry, has no ``image_abspath``, because its image file was never read.
- ``label_url`` and ``label_abspath``: the absolute path or URL of the label and its path
  on the local disk, when ``label_path`` is given. An observation read from the image
  file alone has neither.

A host that reads a label and the image file it describes therefore calls the method
after it has inserted the ``data`` subfield.

For an observation from a row of an index, ``image_path`` is the relative path of the
file as the index gives it, such as the volume and the file specification name joined
with "/", given as a ``str``, and ``label_path`` is None. Only ``filepath``, that string,
and ``basename``, its last component, are set. A caller that wants the file joins the
path with its own root directory. Reading an index never retrieves a data file.

Correcting label text
---------------------

Some product labels hold text that the PDS3 parser cannot read, such as a line of plain
prose outside any comment. The oops readers leave such corrections to the host of the
data set, which overrides the class method ``_correct_label_text(text, filepath)``. It
receives the text of the label, up to its END statement for an attached label, with
<LF> line terminators, and the path of its file, and returns the corrected text. The
default returns the text unchanged, and a host that keeps it has its labels parsed from
their files.

``_read_fileinfo`` calls the method whenever it parses a detached or attached PDS3 label
for its host, before the parser runs. The corrected text is parsed in place of the
file's, and the parsed label keeps the path of its file, so that its pointers find the
files beside it. The labels of an index and of its supplement are parsed as they are
written, without any correction.

The method also sees the labels of other hosts: when
:meth:`~oops.host.Host.from_file` cannot parse a label as written, it tries the
correction of every registered host that has one, as described under `Reading a file`_.
A correction must therefore change only the text that has the defect it corrects and
return any other text as it is.

.. code-block:: python

    class MyCamera(Host):

        @classmethod
        def _correct_label_text(cls, text, filepath):
            return text.replace('\nExtended header missing.\n', '\n')

Reading the rows of an index
----------------------------

A host's ``from_index`` reads its rows through the class method
``_index_rows(filepath, *, supplement=Host.SUPPLEMENT, require_supplement=True,
label=None, row_dicts=None, pds3_options=None, filter=None, validate=None)``, which
returns an iterator over the rows, each a dictionary as described under
`Reading an index`_. Its arguments are those of :meth:`~oops.host.Host.from_index`,
plus these:

``require_supplement``
    True if the supplement must exist. A missing supplement label then raises
    :class:`FileNotFoundError`, and an index whose name gives no supplement name raises
    :class:`~oops.host.HostIndexError`. False reads the index alone in either case. A
    host whose supplement is optional, but that must refuse a missing file the caller
    named, passes ``require_supplement=supplement is not Host.SUPPLEMENT``.

``validate``
    The check of each pair of rows. A column name, or a list or tuple of names, gives
    columns whose values the two rows must share, and two strings are the same if either
    one, in upper case, contains the other. An empty list or tuple skips the comparison
    of columns. A function of the index row and the supplement row returns True if they
    match. By default, the columns that the two tables share are compared, except those
    of the file keys, and two tables that share no column raise
    :class:`~oops.host.HostIndexError`.

The call reads the index label, unless ``label`` is given, checks that the index belongs
to this host, and reads the supplement's label, parsed with the method of
``pds3_options`` alone. With ``row_dicts``, the tables are not read, and the iterator
yields each of the given rows that passes ``filter``. A ``filter`` tests each row as the
iteration reaches it, and a row that lacks a filtered column raises :class:`KeyError`.

How the rows of the index and of its supplement are paired is fixed for each host by
three class attributes:

:attr:`~oops.host.Host.MERGE_SUPPLEMENT_BY_FILE`
    False, the default, pairs each index row with the supplement row in the same
    position, reading the supplement in step with the index. True pairs each index row
    with the supplement row that gives the same file, wherever it is. The supplement is
    then read ahead only as far as that row, and a supplement row read early is held
    until its index row is reached, so a supplement whose rows are k places out of order
    holds at most k rows.

:attr:`~oops.host.Host.INDEX_FILE_KEY` and :attr:`~oops.host.Host.SUPPLEMENT_FILE_KEY`
    The names of the columns whose values, joined with "/", give the file of an index row
    and of a supplement row. Both are ("VOLUME_ID", "FILE_SPECIFICATION_NAME") by
    default, so that a cumulative index that lists one file name under two volumes is
    paired correctly. A host whose tables give the file in other columns names them, for
    example ``("VOLUME_ID", "PATH_NAME", "FILE_NAME")``.

The files of two rows are compared without regard to case, to a leading "/" or "./", or
to the extension of the file name, so that a row that gives a product's label matches a
row that gives its data file.

A host whose supplement is not named after its index overrides the class method
``_supplement_path(index_path)``, which returns the path of the supplement beside the
index, or None if the index's name gives none. In this example, ``_from_row`` stands for
the host's own function that builds the observation of one row:

.. code-block:: python

    class MyCamera(Host):

        MERGE_SUPPLEMENT_BY_FILE = True
        INDEX_FILE_KEY = ('VOLUME_ID', 'PATH_NAME', 'FILE_NAME')

        @classmethod
        def _supplement_path(cls, index_path):
            return index_path.with_name(index_path.name.replace('_raw_image_index',
                                                                '_supplemental_index'))

        @classmethod
        def from_index(cls, filepath, *, supplement=Host.SUPPLEMENT, label=None,
                       row_dicts=None, pds3_options=None, filter=None, **overrides):
            rows = cls._index_rows(filepath, supplement=supplement, label=label,
                                   row_dicts=row_dicts, pds3_options=pds3_options,
                                   filter=filter)
            return (cls._from_row(row, **overrides) for row in rows)

Errors while reading an index
-----------------------------

The call to ``_index_rows`` raises :class:`~oops.host.WrongHostError` for an index of
another host, :class:`~oops.host.PdsHostError` for a label that cannot be parsed,
:class:`~oops.OopsValueError` for an invalid parser option or a label value that the
parser cannot convert, and :class:`FileNotFoundError` for a missing label. It raises
:class:`~oops.host.HostIndexError` for a label that does not describe a readable table,
for a required supplement whose name the index's name does not give, and for tables that
share no column when ``validate`` is not given.

The iteration raises these at the row concerned:

- :class:`FileNotFoundError` when it starts, if a table file does not exist.
- :class:`~oops.host.HostIndexError` for a table file that holds fewer rows than its label
  says, after every row it holds, or more, after the last row.
- :class:`~oops.host.HostIndexError` for a row that is longer than the label's row
  length. The rows are read 100 at a time, so the error is raised at the first of those
  100 rows.
- :class:`~oops.host.HostIndexError` for a pair of rows that give different files, a pair
  that fails ``validate``, or an index row for which the supplement has no row. When the
  rows are paired by file, a missing supplement row is found only once the whole
  supplement has been read, and a supplement row that repeats the file of a row still
  held is reported when it is read.
- :class:`~oops.host.HostIndexError` after the last index row, for supplement rows that
  were not paired with any index row.
- :class:`KeyError` for a row that lacks a column that ``filter`` or a file key names.
- Any error that the host raises as it builds an observation from a row.

Null values
-----------

:func:`~oops.host.is_pds3_null` tells whether a value from a PDS3 label or a VICAR header
is one of the texts that stand for a missing value, listed in
:data:`~oops.host.PDS3_NULL_TEXTS`: "UNK", "N/A", "NULL", "UNKNOWN", "NOT APPLICABLE",
"---", and the empty string. It matches the whole value, without regard to case or to
blanks at either end, so that every host leaves out the same items. Numeric fill values,
such as -999, differ from one data set to another and are left to each host.

Other tools
-----------

The subpackage also provides :func:`~oops.host.read_pds3_image_array` and
:func:`~oops.host.get_fits_image_hdu`, which hosts use to read image arrays, and the
:class:`~oops.host.HostError` exceptions that report a file a host cannot read.

API reference
-------------

.. automodule:: oops.host
    :member-order: bysource
    :members:
    :show-inheritance:
    :exclude-members: __dict__, __hash__, __module__, __weakref__, __annotations__
