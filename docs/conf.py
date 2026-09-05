##########################################################################################
# docs/conf.py: Sphinx configuration for the rms-oops documentation
##########################################################################################

import datetime
import importlib.metadata
import os
import sys

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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Nitpicky mode resolves every type named in a `Parameters:` or `Returns:` block. These
# names have no target to resolve to, so each one is listed here rather than silenced
# globally. Never add a symbol this package owns; give it an API-reference entry instead.
nitpick_ignore = [
    # Napoleon appends ", optional" to the type of an optional parameter. It marks the
    # parameter, not a type, so there is nothing for it to link to.
    ('py:class', 'optional'),
    # Informal type names used by `polymath`, whose docstrings are rendered here because
    # `oops` re-exports its classes. They name a concept, not a class.
    ('py:class', 'array-like'),
    ('py:class', 'convertible'),
    ('py:class', 'number'),
    ('py:class', 'scalar'),
    ('py:class', 'vector-like'),
    # `programs.gold_master` types a callback parameter as "function", which names a
    # concept rather than a class.
    ('py:class', 'function'),
    # `polymath` internals that its own docstrings mention but do not publish.
    ('py:class', 'QubeNDIterator'),
    ('py:class', 'Unit'),
    # `filecache.FCPath`, which has no Sphinx inventory to link to.
    ('py:class', 'FCPath'),
]

# The arithmetic docstrings of `polymath.Qube` cross-reference its operator methods, which
# are not part of the rendered surface here.
nitpick_ignore_regex = [
    (r'py:meth', r'Qube\.__\w+__'),
]

myst_enable_extensions = ['colon_fence', 'deflist']

# Client-side rendering, so no mmdc binary is needed in CI or on ReadTheDocs.
mermaid_output_format = 'raw'

##########################################################################################
