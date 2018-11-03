[![Travis Status](https://travis-ci.org/rock-learning/pytransform.svg?branch=master)](https://travis-ci.org/rock-learning/pytransform)
[![CircleCI Status](https://circleci.com/gh/rock-learning/pytransform/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/rock-learning/pytransform)

# pytransform

A Python library for transformations in three dimensions.

The library focuses on readability and debugging, not on computational
efficiency.
If you want to have an efficient implementation of some function from the
library you can easily extract the relevant code and implement it more
efficiently in a language of your choice.
It makes conversions between rotation and transformation conventions as easy
as possible.

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
pytransform assists in solving these issues. Its documentation clearly
states all of the used conventions, it has various functions to convert
between conventions, it is tightly coupled with Matplotlib to quickly
visualize (or animate) transformations and it is written in Python with
few dependencies. Python is a widely adopted language. It is used in many
domains and supports a wide spectrum of communication to other software.

In addition, pytransform offers...

* the TransformManager which manages complex chains of transformations
* the TransformEditor which allows to modify transformations graphically
  (additionally requires PyQt4)
* the UrdfTransformManager which is able to load transformations from
  [URDF](http://wiki.ros.org/urdf) files (additionally requires
  beautifulsoup4)

pytransform is used in various domains, for example:

* specifying motions of a robot
* learning robot movements from human demonstration
* sensor fusion for human pose estimation

## Installation

Install the package with:

    python setup.py install

## Documentation

The API documentation can be found
[here](https://rock-learning.github.io/pytransform/).

The docmentation of this project can be found in the directory `doc`. To
build the documentation, run e.g. (on unix):

    cd doc
    make html

The HTML documentation is now located at `doc/build/html/index.html`.

## Example

This is just one simple example. You can find more examples in the subfolder
`examples/`.

```python
import numpy as np
import matplotlib.pyplot as plt
import pytransform.rotations as pr
import pytransform.transformations as pt
from pytransform.transform_manager import TransformManager


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

![output](https://rock-learning.github.io/pytransform/_images/plot_transform_manager.png)

## Tests

You can use nosetests to run the tests of this project in the root directory:

    nosetests

A coverage report will be located at `cover/index.html`.
The branch coverage is currently 100% for code that is not related to the
GUI.
