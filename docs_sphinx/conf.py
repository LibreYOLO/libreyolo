# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add libreyolo to path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'LibreYOLO'
copyright = '2024, LibreYOLO Team'
author = 'LibreYOLO Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',          # Support Google-style docstrings
    'sphinx.ext.viewcode',          # Add links to source code
    'sphinx.ext.intersphinx',       # Link to other projects' docs
    'sphinx.ext.autosummary',       # Generate summary tables
    'myst_parser',                  # Write docs in Markdown
]

# MyST Parser settings (allows Markdown)
myst_enable_extensions = [
    "colon_fence",      # ::: directives
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__, __call__',
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# Autosummary
autosummary_generate = True

# Intersphinx - link to external docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'  # Modern, clean theme

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#ff6b35",  # Orange accent
        "color-brand-content": "#ff6b35",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8c5a",
        "color-brand-content": "#ff8c5a",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_title = "LibreYOLO"
html_static_path = ['_static']

# Source file extensions
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Pygments style for code highlighting
pygments_style = 'sphinx'
pygments_dark_style = 'monokai'

