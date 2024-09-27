<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/logo.png" />

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/dfki-ric/pytransform3d/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/dfki-ric/pytransform3d/tree/main)
[![codecov](https://codecov.io/gh/dfki-ric/pytransform3d/branch/main/graph/badge.svg?token=jB10RM3Ujj)](https://codecov.io/gh/dfki-ric/pytransform3d)
[![Paper DOI](http://joss.theoj.org/papers/10.21105/joss.01159/status.svg)](https://doi.org/10.21105/joss.01159)
[![Release DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2553450.svg)](https://doi.org/10.5281/zenodo.2553450)

# pytransform3d

A Python library for transformations in three dimensions.

pytransform3d offers...

* operations like concatenation and inversion for most common representations
  of rotation (orientation) and translation (position)
* conversions between those representations
* clear documentation of transformation conventions
* tight coupling with matplotlib to quickly visualize (or animate)
  transformations
* the TransformManager which manages complex chains of transformations
  (with export to graph visualization as PNG, additionally requires pydot)
* the TransformEditor which allows to modify transformations graphically
  (additionally requires PyQt4/5)
* the UrdfTransformManager which is able to load transformations from
  [URDF](https://wiki.ros.org/urdf) files (additionally requires lxml)
* a matplotlib-like interface to Open3D's visualizer to display and animate
  geometries and transformations (additionally requires Open3D)

pytransform3d is used in various domains, for example:

* specifying motions of a robot
* learning robot movements from human demonstration
* sensor fusion for human pose estimation
* collision detection for robots

The API documentation can be found
[here](https://dfki-ric.github.io/pytransform3d/).

I gave a talk at EuroSciPy 2023 about pytransform3d. Slides are available
[here](https://github.com/AlexanderFabisch/pytransform3d_euroscipy2023/).

## Installation

Use pip to install the package from PyPI:

```bash
pip install pytransform3d[all]
```

or conda:

```bash
conda install -c conda-forge pytransform3d
```

Take a look at the
[installation instructions](https://dfki-ric.github.io/pytransform3d/install.html)
in the documentation for more details.

## Gallery

The following plots and visualizations have been generated with pytransform3d.
The code for most examples can be found in
[the documentation](https://dfki-ric.github.io/pytransform3d/_auto_examples/index.html).

Left: [Nao robot](https://www.softbankrobotics.com/emea/en/nao) with URDF
from [Bullet3](https://github.com/bulletphysics/bullet3).
Right: [Kuka iiwa](https://www.kuka.com/en-de/products/robot-systems/industrial-robots/lbr-iiwa).
The animation is based on pytransform3d's visualization interface to
[Open3D](http://www.open3d.org/).

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/animation_nao.gif" height=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/animation_kuka.gif" height=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/animation_dynamics.gif" height=200px/>

Visualizations based on [Open3D](http://www.open3d.org/).

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/photogrammetry.png" height=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/kuka_trajectories.png" height=200px/>

Various plots based on Matplotlib.

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/example_plot_box.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/cylinders.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/paper/plot_urdf.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/transform_manager_mesh.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/accelerate_cylinder.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/example_plot_screw.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/rotations_axis_angle.png" width=200px/><img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/doc/source/_static/concatenate_uncertain_transforms.png" width=200px/>

Transformation editor based on Qt.

<img src="https://raw.githubusercontent.com/dfki-ric/pytransform3d/main/paper/app_transformation_editor.png" height=300px/>

## Example

This is just one simple example. You can find more examples in the subfolder
`examples/`.

```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


rng = np.random.default_rng(0)

ee2robot = pt.transform_from_pq(
    np.hstack((np.array([0.4, -0.3, 0.5]),
               pr.random_quaternion(rng))))
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

![output](https://dfki-ric.github.io/pytransform3d/_images/sphx_glr_plot_transform_manager_001.png)

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
Execute the following command in the main folder of the repository
to install the dependencies:

```bash
pip install -e .[doc]
```

## Tests

You can use pytest to run the tests of this project in the root directory:

```bash
pytest
```

A coverage report will be located at `htmlcov/index.html`.
Note that you have to install `pytest` to run the tests and `pytest-cov` to
obtain the code coverage report.

## Contributing

If you wish to report bugs, please use the
[issue tracker](https://github.com/dfki-ric/pytransform3d/issues) at
Github. If you would like to contribute to pytransform3d, just open an issue
or a [pull request](https://github.com/dfki-ric/pytransform3d/pulls).
The target branch for pull requests is the develop branch.
The development branch will be merged to main for new releases.
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

It is forbidden to directly push to the main branch. Each new version
has its own development branch from which a pull request will be opened to the
main branch. Only the maintainer of the software is allowed to merge a
development branch to the main branch.

## License

The library is distributed under the
[3-Clause BSD license](https://github.com/dfki-ric/pytransform3d/blob/main/LICENSE).

## Citation

If you use pytransform3d for a scientific publication, I would appreciate
citation of the following paper:

Fabisch, A. (2019). pytransform3d: 3D Transformations for Python.
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
