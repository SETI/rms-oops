##########################################################################################
# docs/conf.py: Sphinx configuration for the rms-oops documentation
##########################################################################################

import datetime
import importlib.metadata
import os
import sys

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles

# Anchored to this file rather than to the working directory, so that autodoc imports the
# same tree whether the build runs from docs/ (as the Makefile does) or from the
# repository root (as the check script and CI do).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(_REPO, 'src'))
sys.path.insert(0, _REPO)                      # for `programs.gold_master`

# -- Project information -----------------------------------------------------

project = 'rms-oops'
copyright = f'{datetime.date.today().year}, SETI Institute'
author = 'SETI Institute'

try:
    release = importlib.metadata.version('rms-oops')
except importlib.metadata.PackageNotFoundError:
    release = '0.0.0'           # a source tree with no install

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.mermaid',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.md']

# The docstrings wrap variable names in single backticks. Napoleon renders the name of
# each entry in a `Parameters:` block in bold, so the default role must be `strong` for a
# mention of that same name in the surrounding prose to match it. Double backticks mark
# code expressions, and italics mark math symbols that are not variable names, such as
# *x*-axis. An API symbol that should link to its own entry carries an explicit role
# instead.
default_role = 'strong'

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
add_module_names = False
autodoc_typehints_format = 'short'

# Applied to every autodoc directive, so that each page names only what differs. A
# directive that sets one of these options overrides the default for that option alone.
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'exclude-members': '__dict__, __hash__, __module__, __weakref__, __annotations__',
}

# Every class documents its constructor arguments in the `__init__` docstring, not in the
# class docstring. With autodoc's default, 'class', that block is dropped and the reader
# sees a signature whose parameters are never explained; 'both' appends the constructor
# docstring to the class description, under the signature autodoc already derives from
# `__init__`.
autoclass_content = 'both'

# Importing oops furnishes SPICE kernels and reads the resource tree named by
# OOPS_RESOURCES, which a documentation builder does not have. These are mocked so that
# autodoc can import every module without it.
autodoc_mock_imports = [
    'cspyce',
    'pylab',
    'matplotlib',
]

# -- Extension configuration -------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
# Render an `Attributes:` block as `:ivar:` fields. With the default directive form,
# Napoleon emits a `py:attribute` for each entry, which collides with the `property`
# of the same name that autodoc already documents.
napoleon_use_ivar = True
# `Properties:` survives only in `programs.gold_master`, which this tree does not
# autodoc; Napoleon does not know the heading, so without this the block would be left
# as raw text and docutils would read the indented descriptions as a definition list
# that unindents unexpectedly. The `oops` and `spicedb` classes use `Attributes:`,
# which Napoleon handles natively.
napoleon_custom_sections = [('Properties', 'params_style')]

# `polymath` is mapped so that a docstring naming one of its symbols in full, such as
# `polymath.Vector3`, links to its own documentation. A bare name resolves against this
# project's inventory instead, which is why `oops.rst` documents the PolyMath classes and
# the aliases of `polymath.typedefs` alongside the classes of `oops` itself.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'polymath': ('https://rms-polymath.readthedocs.io/en/latest/', None),
}

# Nitpicky mode resolves every type named in a `Parameters:` or `Returns:` block. These
# names have no target to resolve to, so each one is listed here rather than silenced
# globally. Never add a symbol this package owns; give it an API-reference entry instead.
nitpick_ignore = [
    # Napoleon appends ", optional" to the type of an optional parameter. It marks the
    # parameter, not a type, so there is nothing for it to link to.
    ('py:class', 'optional'),
    # A plain numeric array, where no PolyMath type is involved. It names a concept
    # rather than a class, so there is nothing for it to link to.
    ('py:class', 'array-like'),
    # `programs.gold_master` types a callback parameter as "function", which names a
    # concept rather than a class.
    ('py:class', 'function'),
    # `polymath` internals that its own docstrings mention but do not publish.
    ('py:class', 'QubeNDIterator'),
    ('py:class', 'Unit'),
    # `filecache.FCPath` and `filecache.FileCache`, which have no Sphinx inventory to
    # link to.
    ('py:class', 'FCPath'),
    ('py:class', 'FileCache'),
    # Each of these is a `polymath.typedefs` `TypeAlias`, so autodoc documents it as a
    # `py:data` object (see `automodule:: polymath.typedefs` in oops.rst); Napoleon,
    # however, renders every `Parameters:`/`Returns:` type as a `py:class` reference, and
    # Sphinx's Python domain never resolves a `class` role against a `data` object. The
    # target exists and is documented; only the role mismatch is unresolvable.
    ('py:class', 'BooleanLike'),
    ('py:class', 'IntValsType'),
    ('py:class', 'MaskType'),
    ('py:class', 'Matrix3Like'),
    ('py:class', 'MatrixLike'),
    ('py:class', 'PairLike'),
    ('py:class', 'QuaternionLike'),
    ('py:class', 'QubeLike'),
    ('py:class', 'ScalarLike'),
    ('py:class', 'ValsType'),
    ('py:class', 'Vector3Like'),
    ('py:class', 'VectorLike'),
]

# The arithmetic docstrings of `polymath.Qube` cross-reference its operator methods, which
# are not part of the rendered surface here.
nitpick_ignore_regex = [
    (r'py:meth', r'Qube\.__\w+__'),
]

myst_enable_extensions = ['colon_fence', 'deflist']

# Client-side rendering, so no mmdc binary is needed in CI or on ReadTheDocs.
mermaid_output_format = 'raw'

# -- The private copy of the API reference -----------------------------------

# The published reference documents the public API alone. The Developer's Guide needs a
# second copy in which the private members are visible too, since a change to `oops`
# means working with them. That copy is built from the same tree with `-t private`, into
# docs/_build/private/html; `scripts/run-all-checks.sh --sphinx` builds both. The tag
# turns on autodoc's `private-members` option, lets Napoleon keep a private member's
# docstring, selects the `.. only:: private` prose that says which copy the reader is
# looking at, and populates the `.. private-only::` blocks of the Developer's Guide. `tags` is a name Sphinx
# binds in this namespace, which ruff cannot see.
_PRIVATE = tags.has('private')                                              # noqa: F821

if _PRIVATE:
    autodoc_default_options['private-members'] = True
    napoleon_include_private_with_doc = True
else:
    # `_BackplaneComparison` is documented only in the private build (see the Developer's
    # Guide's API reference), but public methods of `programs.gold_master` still type a
    # parameter as it in their docstrings. The public build has no target for it.
    nitpick_ignore.append(('py:class', '_BackplaneComparison'))


class _PrivateOnly(Directive):
    """Content that exists only in the private build.

    `.. only:: private` is not enough for the private API pages: its body is parsed in
    every build and merely pruned from the output afterwards, so an autodoc directive
    inside it would still register its objects, and a cross-reference to a private member
    would still be checked, in the public build. This directive parses its body in the
    private build and drops it, unparsed, in the public one.
    """

    has_content = True

    def run(self):
        if not _PRIVATE:
            return []
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, self.content, node, self.content_offset)
        return node.children


def setup(app):
    app.add_directive('private-only', _PrivateOnly)

##########################################################################################
