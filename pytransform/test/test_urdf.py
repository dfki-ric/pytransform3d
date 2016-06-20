import numpy as np
from pytransform.transform_manager import TransformManager
from pytransform.urdf import load_urdf, UrdfException
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_is_instance, assert_raises


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
  <link name="kuka_tcp" />

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

  <joint name="kuka_joint_tcp" type="fixed">
    <parent link="kuka_link_7"/>
    <child link="kuka_tcp"/>
    <origin rpy="0 0 0" xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def test_missing_robot_tag():
    assert_raises(UrdfException, load_urdf, "")


def test_missing_robot_name():
    assert_raises(UrdfException, load_urdf, "<robot/>")


def test_creates_tm():
    tm = load_urdf("<robot name=\"robot_name\"/>")
    assert_is_instance(tm, TransformManager)


def test_missing_link_name():
    assert_raises(UrdfException, load_urdf,
                  "<robot name=\"robot_name\"><link/></robot>")


def test_missing_joint_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint>
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_missing_parent():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_missing_child():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <parent link="link0"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_missing_parent_link_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <parent/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_missing_child_link_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <parent link="link0"/>
        <child/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_reference_to_unknown_child():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <joint name="joint0">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_reference_to_unknown_parent():
    urdf = """
    <robot name="robot_name">
    <link name="link1"/>
    <joint name="joint0">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_missing_joint_type():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, load_urdf, urdf)


def test_without_origin():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    tm = load_urdf(urdf)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(link1_to_link0, np.eye(4))


def test_with_empty_origin():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
        <origin/>
    </joint>
    </robot>
    """
    tm = load_urdf(urdf)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(link1_to_link0, np.eye(4))


def test_with_empty_axis():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="revolute">
        <parent link="link0"/>
        <child link="link1"/>
        <origin/>
        <axis/>
    </joint>
    </robot>
    """
    tm = load_urdf(urdf)
    tm.set_joint("joint0", np.pi)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(
        link1_to_link0,
        np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
    )


def test_ee_frame():
    tm = TransformManager()
    load_urdf(KUKA_LWR_URDF, tm)
    link7_to_link0 = tm.get_transform("kuka_link_7", "kuka_link_0")
    assert_array_almost_equal(
        link7_to_link0,
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1.261],
                  [0, 0, 0, 1]])
    )


def test_joint_angles():
    tm = load_urdf(KUKA_LWR_URDF)
    for i in range(1, 8):
        tm.set_joint("kuka_joint_%d" % i, 0.1 * i)
    link7_to_link0 = tm.get_transform("kuka_link_7", "kuka_link_0")
    assert_array_almost_equal(
        link7_to_link0,
        np.array([[-0.037301, -0.977762, 0.206374, 0.03205],
                  [0.946649, 0.031578, 0.320715, -0.018747],
                  [-0.3201, 0.207327, 0.92442, 1.23715],
                  [0., 0., 0., 1.]])
    )


def test_fixed_joint():
    tm = TransformManager()
    load_urdf(KUKA_LWR_URDF, tm)
    tcp_to_link0 = tm.get_transform("kuka_tcp", "kuka_link_0")
    assert_array_almost_equal(
        tcp_to_link0,
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1.311],
                  [0, 0, 0, 1]])
    )
