[![Travis Status](https://travis-ci.org/rock-learning/pytransform3d.svg?branch=master)](https://travis-ci.org/rock-learning/pytransform3d)
[![CircleCI Status](https://circleci.com/gh/rock-learning/pytransform3d/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/rock-learning/pytransform3d)
[![Paper DOI](http://joss.theoj.org/papers/10.21105/joss.01159/status.svg)](https://doi.org/10.21105/joss.01159)
[![Release DOI](https://zenodo.org/badge/91809394.svg)](https://zenodo.org/badge/latestdoi/91809394)

# pytransform3d

A Python library for transformations in three dimensions.

The library focuses on readability and debugging, not on computational
efficiency.
If you want to have an efficient implementation of some function from the
library you can easily extract the relevant code and implement it more
efficiently in a language of your choice.

The library integrates well with the
[scientific Python ecosystem](https://scipy-lectures.org/)
with its core libraries Numpy, Scipy and Matplotlib.
We rely on [Numpy](https://numpy.org/) for linear algebra and on
[Matplotlib](https://matplotlib.org/) to offer plotting functionalities.
[Scipy](https://scipy.org/scipylib/index.html) is used if you want to
automatically compute new transformations from a graph of existing
transformations.

Heterogenous software systems that consist of proprietary and open source
software are often combined when we work with transformations.
For example, suppose you want to transfer a trajectory demonstrated by a human
to a robot. The human trajectory could be measured from an RGB-D camera, fused
with IMU sensors that are attached to the human, and then translated to
joint angles by inverse kinematics. That involves at least three different
software systems that might all use different conventions for transformations.
Sometimes even one software uses more than one convention.
The following aspects are of crucial importance to glue and debug
transformations in systems with heterogenous and often incompatible
software:
* Compatibility: Compatibility between heterogenous softwares is a difficult
  topic. It might involve, for example, communicating between proprietary and
  open source software or different languages.
* Conventions: Lots of different conventions are used for transformations
  in three dimensions. These have to be determined or specified.
* Conversions: We need conversions between these conventions to
  communicate transformations between different systems.
* Visualization: Finally, transformations should be visually verified
  and that should be as easy as possible.

pytransform3d assists in solving these issues. Its documentation clearly
states all of the used conventions, it makes conversions between rotation
and transformation conventions as easy as possible, it is tightly coupled
with Matplotlib to quickly visualize (or animate) transformations and it
is written in Python with few dependencies. Python is a widely adopted
language. It is used in many domains and supports a wide spectrum of
communication to other software.

In addition, pytransform3d offers...

* the TransformManager which manages complex chains of transformations
  (with export to graph visualization as PNG, additionally requires pydot)
* the TransformEditor which allows to modify transformations graphically
  (additionally requires PyQt4 or PyQt5)
* the UrdfTransformManager which is able to load transformations from
  [URDF](http://wiki.ros.org/urdf) files (additionally requires
  beautifulsoup4)

pytransform3d is used in various domains, for example:

* specifying motions of a robot
* learning robot movements from human demonstration
* sensor fusion for human pose estimation

## Installation

Use pip to install the package from PyPI:

    [sudo] pip[3] install [--user] pytransform3d[all,doc,test]

You can install pytransform3d[all] if you want to have support for pydot
export. Make sure to install graphviz (on Ubuntu: `sudo apt install graphviz`)
if you want to use this feature. If you want to have support for the Qt GUI
you have to install PyQt 4 or 5 (on Ubuntu: `sudo apt install python3-pyqt5`;
conda: `conda install pyqt`).

You can also install from the current git version: clone the repository and go
to the main folder. Install dependencies with:

    pip install -r requirements.txt

Install the package with:

    python setup.py install

Also pip supports installation from a git repository:

    pip install git+https://github.com/rock-learning/pytransform3d.git

## Documentation

The API documentation can be found
[here](https://rock-learning.github.io/pytransform3d/).

The docmentation of this project can be found in the directory `doc`.
To build the documentation, run e.g. (on unix):

    cd doc
    make html

The HTML documentation is now located at `doc/build/html/index.html`.
You need the following packages to build the documentation:

    pip install numpydoc sphinx sphinx-gallery sphinx-bootstrap-theme

## Example

This is just one simple example. You can find more examples in the subfolder
`examples/`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


random_state = np.random.RandomState(0)

ee2robot = pt.transform_from_pq(
    np.hstack((np.array([0.4, -0.3, 0.5]),
               pr.random_quaternion(random_state))))
cam2robot = pt.transform_from_pq(
    np.hstack((np.array([0.0, 0.0, 0.8]), pr.q_id)))
object2cam = pt.transform_from(
    pr.active_matrix_from_intrinsic_euler_xyz(np.array([0.0, 0.0, -0.5])),
    np.array([0.5, 0.1, 0.1]))

tm = TransformManager()
tm.add_transform("end-effector", "robot", ee2robot)
tm.add_transform("camera", "robot", cam2robot)
tm.add_transform("object", "camera", object2cam)

ee2object = tm.get_transform("end-effector", "object")

ax = tm.plot_frames_in("robot", s=0.1)
ax.set_xlim((-0.25, 0.75))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((0.0, 1.0))
plt.show()
```

![output](https://rock-learning.github.io/pytransform3d/_images/plot_transform_manager.png)

## Gallery

The following plots and visualizations have been generated with pytransform3d.

Left: [Nao robot](https://www.softbankrobotics.com/emea/en/nao) with URDF
from [Bullet3](https://github.com/bulletphysics/bullet3).
Right: [Kuka iiwa](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/lbr-iiwa).
The animation is based on pytransform3d's visualization interface to
[Open3D](http://www.open3d.org/).

<img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/animation_nao.gif" height=400px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/animation_kuka.gif" height=400px/>

Various plots based on Matplotlib.

<img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/example_plot_box.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/cylinders.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/paper/plot_urdf.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/transform_manager_mesh.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/accelerate_cylinder.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/example_plot_screw.png" width=300px/><img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/doc/source/_static/rotations_axis_angle.png" width=300px/>

Transformation editor based on Qt.

<img src="https://raw.githubusercontent.com/rock-learning/pytransform3d/master/paper/app_transformation_editor.png" height=300px/>

## Tests

You can use nosetests to run the tests of this project in the root directory:

    nosetests

A coverage report will be located at `cover/index.html`.
Note that you have to install `nose` to run the tests and `coverage` to obtain
the code coverage report.
The branch coverage is currently 100% for code that is not related to the
GUI.

## Contributing

If you wish to report bugs, please use the issue tracker at Github.
If you would like to contribute to pytransform3d, just open an issue or
a merge request.
