[![Travis Status](https://travis-ci.org/rock-learning/pytransform3d.svg?branch=master)](https://travis-ci.org/rock-learning/pytransform3d)
[![CircleCI Status](https://circleci.com/gh/rock-learning/pytransform3d/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/rock-learning/pytransform3d)

# pytransform3d

A Python library for transformations in three dimensions.

The library focuses on readability and debugging, not on computational
efficiency.
If you want to have an efficient implementation of some function from the
library you can easily extract the relevant code and implement it more
efficiently in a language of your choice.

The library integrates well with the
[scientific Python ecosystem](https://www.scipy-lectures.org/)
with its core libraries Numpy, Scipy and Matplotlib.
We rely on [Numpy](https://www.numpy.org/) for linear algebra and on
[Matplotlib](https://matplotlib.org/) to offer plotting functionalities.
[Scipy](https://www.scipy.org/) is used if you want to automatically
compute new transformations from a graph of existing transformations.

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
  (additionally requires PyQt4)
* the UrdfTransformManager which is able to load transformations from
  [URDF](http://wiki.ros.org/urdf) files (additionally requires
  beautifulsoup4)

pytransform3d is used in various domains, for example:

* specifying motions of a robot
* learning robot movements from human demonstration
* sensor fusion for human pose estimation

## Installation

Use pip to install the package:

    [sudo] pip[3] install [--user] pytransform3d

You can install pytransform3d[all] if you want to have support for pydot
export. Make sure to install graphviz (on Ubuntu: `sudo apt install graphviz`)
if you want to use this feature.

... or clone the repository and go to the main folder.

Install dependencies with:

    pip install -r requirements.txt

Install the package with:

    python setup.py install

## Documentation

The API documentation can be found
[here](https://rock-learning.github.io/pytransform3d/).

The docmentation of this project can be found in the directory `doc`.
Note that currently sphinx 1.6.7 is required to build the documentation.
To build the documentation, run e.g. (on unix):

    cd doc
    make html

The HTML documentation is now located at `doc/build/html/index.html`.
Note that `sphinx` is required to build the documentation.

## Example

This is just one simple example. You can find more examples in the subfolder
`examples/`.

```python
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager


random_state = np.random.RandomState(0)

ee2robot = pt.transform_from_pq(
    np.hstack((np.array([0.4, -0.3, 0.5]), pr.random_quaternion(random_state))))
cam2robot = pt.transform_from_pq(
    np.hstack((np.array([0.0, 0.0, 0.8]), pr.q_id)))
object2cam = pt.transform_from(
    pr.matrix_from_euler_xyz(np.array([0.0, 0.0, 0.5])),
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
