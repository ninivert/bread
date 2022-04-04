# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'bread'
copyright = '2022, G. BRENNA, F. MASSARD, N. VADOT'
author = 'G. BRENNA, F. MASSARD, N. VADOT'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	# 'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
]

extensions.append('autoapi.extension')

autoapi_type = 'python'
autoapi_dirs = [
	'../../src/bread'
	# '../../src/bread/algo/lineage',
	# '../../src/bread/data',
]
autoapi_ignore = [
	'**.ipynb_checkpoints**',
	'**__pycache__**',
	'**bread/vis**',
	'**bread/gui**',
	'**bread/cli**',
	'**bread/algo/tracking**'
]
autoapi_member_order = 'bysource'
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members']

# notebooks
extensions.append('nbsphinx')
jupyter_execute_notebooks = "off"

# extensions.append('sphinx.ext.autosummary')
# autosummary_generate = True  # Turn on sphinx.ext.autosummary

# extensions.append('sphinxcontrib.apidoc')
# apidoc_module_dir = '../../src/bread'
# apidoc_excluded_paths = [
# 	'../../src/bread/vis',
# 	'../../src/bread/algo/tracking',  # TODO
# 	'../../src/bread/cli',
# 	'../../src/bread/gui'
# ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['examples/data', 'examples/datamanip', 'examples/segmentation', 'examples/tracker']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']