.. pytransform3d documentation master file, created by
   sphinx-quickstart on Thu Nov 20 21:01:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
pytransform3d
=============

.. raw:: html

    <div class="container-fluid">
      <div class="row">


        <div class="col-md-4">
          <div class="panel panel-default">
            <div class="panel-heading">
              <h3 class="panel-title">Contents</h3>
            </div>
            <div class="panel-body">

.. toctree::
   :maxdepth: 1

   install
   notation
   rotations
   transformations
   transformation_ambiguities
   euler_angles
   transformation_modeling
   transform_manager
   camera
   animations
   api
   _auto_examples/index

.. raw:: html

            </div>
          </div>
        </div>

        <div class="col-md-8">

pytransform3d covers the following groups of transformations.
This documentation explains how you can work with pytransform3d and with
3D transformations in general.

+-------+--------------------+-----------------------------------------+
| Group | Description        | Representations                         |
+=======+====================+=========================================+
| SO(3) | 3D rotations       | unit quaternion, rotation matrix,       |
|       |                    | axis-angle, Euler angles, rotor         |
+-------+--------------------+-----------------------------------------+
| SE(3) | 3D transformations | transformation matrix, translation      |
|       | (rotation and      | and unit quaternion, exponential        |
|       | translation)       | coordinates, logarithm of               |
|       |                    | transformation, unit dual quaternion    |
+-------+--------------------+-----------------------------------------+


-----
Scope
-----

The library focuses on readability and debugging, not on computational
efficiency. If you want to have an efficient implementation of some function
from the library you can easily extract the relevant code and implement it
more efficiently in a language of your choice.

The library integrates well with the
`scientific Python ecosystem <https://scipy-lectures.org/>`_
with its core libraries Numpy, Scipy and Matplotlib.
We rely on `Numpy <https://numpy.org/>`_ for linear algebra and on
`Matplotlib <https://matplotlib.org/>`_ to offer plotting functionalities.
`Scipy <https://scipy.org/scipylib/index.html>`_ is used if you want to
automatically compute transformations from a graph of transformations.

pytransform3d offers...

* operations for most common representations of rotation (orientation) and
  translation (position)
* conversions between those representations
* clear documentation of transformation conventions
* tight coupling with matplotlib to quickly visualize (or animate)
  transformations
* the TransformManager which manages complex chains of transformations
* the TransformEditor which allows to modify transformations graphically
* the UrdfTransformManager which is able to load transformations from
  `URDF <https://wiki.ros.org/urdf>`_ files
* a matplotlib-like interface to Open3D's visualizer to display
  geometries and transformations


--------
Citation
--------

If you use pytransform3d for a scientific publication, I would appreciate
citation of the following paper:

Fabisch, (2019). pytransform3d: 3D Transformations for Python.
Journal of Open Source Software, 4(33), 1159, |DOI|_

.. |DOI| image:: http://joss.theoj.org/papers/10.21105/joss.01159/status.svg
.. _DOI: https://doi.org/10.21105/joss.01159

.. raw:: html

        </div>

      </div>
    </div>

