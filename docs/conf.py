##########################################################################################
# docs/conf.py: Sphinx configuration for the rms-oops documentation
##########################################################################################

import datetime
import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))
sys.path.insert(0, os.path.abspath('..'))      # for `programs.gold_master`

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

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
add_module_names = False
autodoc_typehints_format = 'short'

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

myst_enable_extensions = ['colon_fence', 'deflist']

# Client-side rendering, so no mmdc binary is needed in CI or on ReadTheDocs.
mermaid_output_format = 'raw'

##########################################################################################
