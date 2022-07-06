import datetime

import tophu

# Project information.
project = "tophu"
release = version = tophu.__version__
author = 'California Institute of Technology ("Caltech")'
copyright = f"{datetime.date.today().year}, {author}"

# General configuration.
extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]
templates_path = ["templates"]

# Options for HTML output.
html_theme = "sphinx_rtd_theme"
html_title = "tophu"

# Extension configuration options: `sphinxcontrib-bibtex`.
bibtex_bibfiles = ["references.bib"]
bibtex_footbibliography_header = ".. rubric:: References"

# Extension configuration options: `sphinx.ext.autodoc`.
autodoc_typehints = "none"

# Extension configuration options: `sphinx.ext.autosummary`.
autosummary_generate = True

# Extension configuration options: `sphinx-copybutton`.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Extension configuration options: `sphinx.ext.napoleon`.
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = False
