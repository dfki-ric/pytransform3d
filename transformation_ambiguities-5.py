import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import random_quaternion, q_id
from pytransform3d.transformations import transform_from_pq
from pytransform3d.transform_manager import TransformManager


random_state = np.random.RandomState(0)

camera2body = transform_from_pq(
    np.hstack((np.array([0.4, -0.3, 0.5]),
               random_quaternion(random_state))))
object2camera = transform_from_pq(
    np.hstack((np.array([0.0, 0.0, 0.3]),
               random_quaternion(random_state))))

tm = TransformManager()
tm.add_transform("camera", "body", camera2body)
tm.add_transform("object", "camera", object2camera)

ax = tm.plot_frames_in("body", s=0.1)
ax.set_xlim((-0.15, 0.65))
ax.set_ylim((-0.4, 0.4))
ax.set_zlim((0.0, 0.8))
plt.show()