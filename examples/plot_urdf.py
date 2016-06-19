import matplotlib.pyplot as plt
from pytransform.urdf import load_urdf


KUKA_LWR_URDF = """
<?xml version="1.0" ?>
<robot name="kuka_lwr">
  <link name="kuka_link_0" />
  <link name="kuka_link_1" />
  <link name="kuka_link_2" />
  <link name="kuka_link_3" />
  <link name="kuka_link_4" />
  <link name="kuka_link_5" />
  <link name="kuka_link_6" />
  <link name="kuka_link_7" />

  <joint name="kuka_joint_1" type="revolute">
    <parent link="kuka_link_0"/>
    <child link="kuka_link_1"/>
    <origin rpy="0 0 0" xyz="0 0 0.1575"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_2" type="revolute">
    <parent link="kuka_link_1"/>
    <child link="kuka_link_2"/>
    <origin rpy="1.57079632679   0 3.14159265359" xyz="0 0 0.2025"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_3" type="revolute">
    <parent link="kuka_link_2"/>
    <child link="kuka_link_3"/>
    <origin rpy="1.57079632679 0 3.14159265359" xyz="0 0.2045 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_4" type="revolute">
    <parent link="kuka_link_3"/>
    <child link="kuka_link_4"/>
    <origin rpy="1.57079632679 0 0" xyz="0 0 0.2155"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_5" type="revolute">
    <parent link="kuka_link_4"/>
    <child link="kuka_link_5"/>
    <origin rpy="-1.57079632679 3.14159265359 0" xyz="0 0.1845 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_6" type="revolute">
    <parent link="kuka_link_5"/>
    <child link="kuka_link_6"/>
    <origin rpy="1.57079632679 0 0" xyz="0 0 0.2155"/>
    <axis xyz="0 0 1"/>
  </joint>

  <joint name="kuka_joint_7" type="revolute">
    <parent link="kuka_link_6"/>
    <child link="kuka_link_7"/>
    <origin rpy="-1.57079632679 3.14159265359 0" xyz="0 0.081 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

tm = load_urdf(KUKA_LWR_URDF)
tm.plot_frames_in("kuka_lwr", s=0.05, show_name=True)
plt.show()
