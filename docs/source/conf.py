import datetime

import tophu

# Project information.
project = "tophu"
release = version = tophu.__version__
author = 'California Institute of Technology ("Caltech")'
copyright = f"{datetime.date.today().year}, {author}"

# General configuration.
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

# Options for HTML output.
html_theme = "sphinx_rtd_theme"
html_title = "tophu"

# Extension configuration options: `sphinxcontrib-bibtex`.
bibtex_bibfiles = ["references.bib"]

# Extension configuration options: `sphinx.ext.autosummary`.
autosummary_generate = True
autosummary_generate_overwrite = False

# Extension configuration options: `sphinx-copybutton`.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Extension configuration options: `sphinx.ext.napoleon`.
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = False
