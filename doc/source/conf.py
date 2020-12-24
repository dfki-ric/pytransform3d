# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_bootstrap_theme

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    #"sphinx.ext.mathjax",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

autodoc_default_options = {"member-order": "bysource"}
autosummary_generate = True  # generate files at doc/source/_apidoc
class_members_toctree = False
numpydoc_show_class_members = False

# class template from https://stackoverflow.com/a/62613202/915743
templates_path = ["_templates"]
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

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None)
}
intersphinx_timeout = 10


# experimental API, works with sphinx-gallery 0.8.2
class Open3DScraper(object):
    def __repr__(self):
        return "Open3DScraper"

    def __call__(self, block, block_vars, gallery_conf, **kwargs):
        """Scrape Open3D images.

        Parameters
        ----------
        block : tuple
            A tuple containing the (label, content, line_number) of the block.
        block_vars : dict
            Dict of block variables.
        gallery_conf : dict
            Contains the configuration of Sphinx-Gallery
        **kwargs : dict
            Additional keyword arguments to pass to
            :meth:`~matplotlib.figure.Figure.savefig`, e.g. ``format='svg'``.
            The ``format`` kwarg in particular is used to set the file extension
            of the output file (currently only 'png', 'jpg', and 'svg' are
            supported).

        Returns
        -------
        rst : str
            The ReSTructuredText that will be rendered to HTML containing
            the images.
        """
        import glob
        import shutil
        from sphinx_gallery.scrapers import figure_rst

        path_current_example = os.path.dirname(block_vars['src_file'])
        jpgs = sorted(glob.glob(os.path.join(
            path_current_example, "__open3d_rendered_image.jpg")))

        image_names = list()
        image_path_iterator = block_vars["image_path_iterator"]
        for jpg in jpgs:
            this_image_path = image_path_iterator.next()
            image_names.append(this_image_path)
            shutil.move(jpg, this_image_path)
        return figure_rst(image_names, gallery_conf["src_dir"])

        with open("test.log", "w") as f:
            f.write(str(gallery_conf))
            f.write(str(block_vars["image_path_iterator"]))
            images = glob.glob("__open3d_rendered_image_*.jpg")
            for image in images:
                f.write(str("image: %s" % image))
                os.remove(image)
        rst = ""
        return rst


from sphinx_gallery.sorting import ExplicitOrder


sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "_auto_examples",
    "subsection_order": ExplicitOrder([
        "../../examples/plots", "../../examples/animations",
        "../../examples/visualizations", "../../examples/apps"]),
    "reference_url": {"pytransform3d": None},
    "filename_pattern": "/(?:plot|animate|vis)_",
    "image_scrapers": ("matplotlib", Open3DScraper()),
}
