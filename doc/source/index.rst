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

+-------+--------------------+-----------------------------------------+
| Group | Description        | Representations                         |
+=======+====================+=========================================+
| SO(3) | 3D rotations       | unit quaternion, rotation matrix,       |
|       |                    | axis-angle, Euler angles                |
+-------+--------------------+-----------------------------------------+
| SE(3) | 3D transformations | transformation matrix, translation      |
|       | (rotation and      | and unit quaternion, exponential        |
|       | translation)       | coordinates, logarithm of               |
|       |                    | transformation                          |
+-------+--------------------+-----------------------------------------+

------------
Installation
------------

pytransform3d is available at
`GitHub <https://github.com/rock-learning/pytransform3d>`_.
The readme there contains installation instructions. This documentation
explains how you can work with pytransform3d and with 3D transformations
in general.

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

