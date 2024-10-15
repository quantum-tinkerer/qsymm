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

package_path = os.path.abspath("../..")
# Insert into sys.path so that we can import qsymm here
sys.path.insert(0, package_path)

import qsymm # noqa: E402

# Insert into PYTHONPATH so that jupyter-sphinx will pick it up
os.environ["PYTHONPATH"] = ":".join((package_path, os.environ.get("PYTHONPATH", "")))

# -- Project information -----------------------------------------------------

project = 'Qsymm'
copyright = '2019, Qsymm Authors'
author = 'Qsymm Authors'
version = qsymm.__version__
release = qsymm.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "jupyter_sphinx",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# text in backticks should be treated as references to the API documentation
default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The license file has no extension, so Sphinx ignores it by default, so we
# must add it here
html_extra_path = ['LICENSE']

# -- Extension config

autosummary_generate = True
