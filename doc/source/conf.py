# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('sphinxext'))

import sphinx_bootstrap_theme

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    #'sphinx.ext.mathjax',
    'sphinx.ext.imgmath',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []
exclude_trees = ["_templates", "sphinxext"]
source_suffix = '.rst'
source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'pytransform3d'
copyright = u'2014-2020, Alexander Fabisch, DFKI GmbH, Robotics Innovation Center'
version = __import__("pytransform3d").__version__
release = __import__("pytransform3d").__version__
language = 'en'
today_fmt = '%B %d, %Y'
add_function_parentheses = True
show_authors = True
pygments_style = 'sphinx'
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {'bootswatch_theme': "readable"}
html_static_path = ['_static']
html_last_updated_fmt = '%b %d, %Y'
html_use_smartypants = True
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']
# Seems to be fixed now and only applies to old versions:
# The problem is that the numpydoc doc scraper is building
# autosummary documentation after autosummary has already run.
# To avoid warnings, we have to set this to variable.
# Source: http://stackoverflow.com/questions/12206334
#numpydoc_show_class_members = True

intersphinx_mapping = {'http://docs.python.org/': None}
