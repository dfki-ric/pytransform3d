import matplotlib.pyplot as plt
from urdf import UrdfTransformManager


URDF = """
<?xml version="1.0"?>
  <robot name="collision">
    <link name="collision">
      <visual name="collision_01">
        <origin xyz="0 0 1" rpy="1 0 0"/>
        <geometry>
          <sphere radius="1"/>
          <cylinder radius="0.5" length="2"/>
          <box size="1 1 1"/>
        </geometry>
      </visual>
  </robot>
"""


if __name__ == "__main__":
    tm = UrdfTransformManager()
    tm.load_urdf(URDF)
    ax = tm.plot_frames_in("collision", s=0.1)
    tm.plot_collision_objects("collision", ax)
    plt.show()
