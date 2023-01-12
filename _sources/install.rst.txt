============
Installation
============

You can install pytransform3d with pip:

.. code-block:: bash

    pip install pytransform3d

or conda (since version 1.8):

.. code-block:: bash

    conda install -c conda-forge pytransform3d

(More detailed instructions
`here <https://github.com/conda-forge/pytransform3d-feedstock#installing-pytransform3d>`_.)

---------------------
Optional Dependencies
---------------------

When using pip, you can install pytransform3d with the options all, doc, and
test.

* `all` will add support for loading meshes, the 3D visualizer of
  pytransform3d, and pydot export of `TransformManager` objects.
* `doc` will install necessary dependencies to build this documentation.
* `test` will install dependencies to run the unit tests.

For example, you can call

.. code-block:: bash

    python -m pip install pytransform3d[all]

Unfortunately, pip cannot install all dependencies:

* If you want to have support for pydot export of `TransformManager` objects,
  make sure to install graphviz (on Ubuntu: `sudo apt install graphviz`) if
  you want to use this feature.
* If you want to have support for the Qt GUI you have to install PyQt 4 or 5
  (on Ubuntu: `sudo apt install python3-pyqt5`; conda: `conda install pyqt`).

------------------------
Installation from Source
------------------------

You can also install from the current git version. pytransform3d is available
at `GitHub <https://github.com/dfki-ric/pytransform3d>`_. Clone the repository
and go to the main folder. Install dependencies with:

.. code-block:: bash

    pip install -r requirements.txt

Install the package with:

.. code-block:: bash

    python setup.py install

pip also supports installation from a git repository:

.. code-block:: bash

    pip install git+https://github.com/dfki-ric/pytransform3d.git
