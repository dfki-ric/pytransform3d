# pytransform

[![Travis Status](https://travis-ci.org/rock-learning/pytransform.svg?branch=master)](https://travis-ci.org/rock-learning/pytransform)
[![CircleCI Status](https://circleci.com/gh/rock-learning/pytransform/tree/master.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/rock-learning/pytransform)

A Python library for transformations in three dimensions. It makes conversions
between rotation and transformation conventions as easy as possible. The
library focuses on readability and debugging, not on computational efficiency.
If you want to have an efficient implementation of some function from the
library you can easily extract the relevant code and implement it more
efficiently in a language of your choice.

pytransform relies on NumPy for linear algebra and on Matplotlib to offer
plotting functionalities.

In addition, pytransform offers...

* the TransformManager which manages complex chains of transformations
* the TransformEditor which allows to modify transformations graphically
  (requires PyQt4)
* the UrdfTransformManager which is able to load transformations from
  [URDF](http://wiki.ros.org/urdf) files (requires beautifulsoup4)

All of these tools can be easily integrated in any Python program.

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
