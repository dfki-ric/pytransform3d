[![CircleCI Status](https://circleci.com/gh/dfki-ric/pytransform3d/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/dfki-ric/pytransform3d)
[![codecov](https://codecov.io/gh/dfki-ric/pytransform3d/branch/master/graph/badge.svg?token=jB10RM3Ujj)](https://codecov.io/gh/dfki-ric/pytransform3d)
[![Paper DOI](http://joss.theoj.org/papers/10.21105/joss.01159/status.svg)](https://doi.org/10.21105/joss.01159)
[![Release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2553450.svg)](https://doi.org/10.5281/zenodo.2553450)

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
states all of the used conventions, it aims at making conversions between
rotation and transformation conventions as easy as possible, it is tightly
coupled with Matplotlib to quickly visualize (or animate) transformations and
it is written in Python with few dependencies. Python is a widely adopted
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

```bash
[sudo] pip[3] install [--user] pytransform3d[all,doc,test]
```

You can install pytransform3d[all] if you want to have support for pydot
export. Make sure to install graphviz (on Ubuntu: `sudo apt install graphviz`)
if you want to use this feature. If you want to have support for the Qt GUI
you have to install PyQt 4 or 5 (on Ubuntu: `sudo apt install python3-pyqt5`;
conda: `conda install pyqt`).

You can also install from the current git version: clone the repository and go
to the main folder. Install dependencies with:

```bash
pip install -r requirements.txt
```

Install the package with:

```bash
python setup.py install
```

Also pip supports installation from a git repository:

```bash
pip install git+https://github.com/dfki-ric/pytransform3d.git
```

Since version 1.8 you can also install pytransform3d with conda from
conda-forge. See
[here](https://github.com/conda-forge/pytransform3d-feedstock#installing-pytransform3d)
for instructions.

## Documentation

The API documentation can be found
[here](https://dfki-ric.github.io/pytransform3d/).

The documentation can be found in the directory `doc`.
To build the documentation, run e.g. (on linux):

```bash
cd doc
make html
```

The HTML documentation is now located at `doc/build/html/index.html`.
You need the following packages to build the documentation:

```bash
pip install numpydoc sphinx sphinx-gallery sphinx-bootstrap-theme
```

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

![output](https://dfki-ric.github.io/pytransform3d/_images/plot_transform_manager.png)

## Gallery

The following plots and visualizations have been generated with pytransform3d.
The code for most examples can be found in
[the documentation](https://dfki-ric.github.io/pytransform3d/_auto_examples/index.html).

Left: [Nao robot](https://www.softbankrobotics.com/emea/en/nao) with URDF
from [Bullet3](https://github.com/bulletphysics/bullet3).
Right: [Kuka iiwa](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/lbr-iiwa).
The animation is based on pytransform3d's visualization interface to
[Open3D](http://www.open3d.org/).

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/animation_nao.gif" height=400px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/animation_kuka.gif" height=400px/>

Visualizations based on [Open3D](http://www.open3d.org/).

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/photogrammetry.png" height=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/kuka_trajectories.png" height=300px/>

Various plots based on Matplotlib.

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/example_plot_box.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/cylinders.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/paper/plot_urdf.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/transform_manager_mesh.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/accelerate_cylinder.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/example_plot_screw.png" width=300px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/doc/source/_static/rotations_axis_angle.png" width=300px/>

Transformation editor based on Qt.

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/master/paper/app_transformation_editor.png" height=300px/>

## Tests

You can use nosetests to run the tests of this project in the root directory:

    nosetests

A coverage report will be located at `cover/index.html`.
Note that you have to install `nose` to run the tests and `coverage` to obtain
the code coverage report.

## Contributing

If you wish to report bugs, please use the
[issue tracker](https://github.com/dfki-ric/pytransform3d/issues) at
Github. If you would like to contribute to pytransform3d, just open an issue
or a [pull request](https://github.com/dfki-ric/pytransform3d/pulls).
The target branch for pull requests is the develop branch.
The development branch will be merged to master for new releases.
If you have questions about the software, you should ask them in the
[discussion section](https://github.com/dfki-ric/pytransform3d/discussions).

The recommended workflow to add a new feature, add documentation, or fix a bug
is the following:

* Push your changes to a branch (e.g. `feature/x`, `doc/y`, or `fix/z`) of your
  fork of the pytransform3d repository.
* Open a pull request to the latest development branch. There is usually an
  open merge request from the latest development branch to the main branch.
* When the latest development branch is merged to the main branch, a new
  release will be made.

Note that there is a
[checklist](https://github.com/dfki-ric/pytransform3d/wiki#checklist-for-new-features)
for new features.

It is forbidden to directly push to the main branch (master). Each new version
has its own development branch from which a pull request will be opened to the
main branch. Only the maintainer of the software is allowed to merge a
development branch to the main branch.

## Citation

If you use pytransform3d for a scientific publication, I would appreciate
citation of the following paper:

Fabisch, (2019). pytransform3d: 3D Transformations for Python.
Journal of Open Source Software, 4(33), 1159,
[![Paper DOI](http://joss.theoj.org/papers/10.21105/joss.01159/status.svg)](https://doi.org/10.21105/joss.01159)

Bibtex entry:

```bibtex
@article{Fabisch2019,
  doi = {10.21105/joss.01159},
  url = {https://doi.org/10.21105/joss.01159},
  year = {2019},
  publisher = {The Open Journal},
  volume = {4},
  number = {33},
  pages = {1159},
  author = {Alexander Fabisch},
  title = {pytransform3d: 3D Transformations for Python},
  journal = {Journal of Open Source Software}
}
```
