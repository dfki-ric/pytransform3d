import numpy as np
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from, concat
from .rotations import matrix_from_axis_angle


class TransformManagerWithJoints(TransformManager):
    def __init__(self):
        super(TransformManagerWithJoints, self).__init__()
        self.default_transformations = {}

    def add_joint(self, joint_name, from_frame, to_frame, A2B, axis):
        self.add_transform(from_frame, to_frame, A2B)
        if joint_name is not None:
            self.default_transformations[joint_name] = (from_frame, to_frame,
                                                        A2B, axis)

    def set_joint(self, joint_name, angle):
        from_frame, to_frame, A2B, axis = self.default_transformations[joint_name]
        joint_rotation = matrix_from_axis_angle(np.hstack((axis, [angle])))
        joint2A = transform_from(joint_rotation, np.zeros(3))
        self.add_transform(from_frame, to_frame, concat(joint2A, A2B))


class Node(object):
    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.A2B = np.eye(4)
        self.joint_name = None
        self.joint_axis = None


def matrix_from_rpy(rpy):
    # fixed axis roll, pitch and yaw angles
    roll, pitch, yaw = rpy
    ca1 = np.cos(yaw)
    sa1 = np.sin(yaw)
    cb1 = np.cos(pitch)
    sb1 = np.sin(pitch)
    cc1 = np.cos(roll)
    sc1 = np.sin(roll)
    R = np.array([
        [ca1 * cb1, ca1 * sb1 * sc1 - sa1 * cc1, ca1 * sb1 * cc1 + sa1 * sc1],
        [sa1 * cb1, sa1 * sb1 * sc1 + ca1 * cc1, sa1 * sb1 * cc1 - ca1 * sc1],
        [-sb1, cb1 * sc1, cb1 * cc1]
    ])
    return R


def load_urdf(urdf_xml, tm=None):
    if tm is not None and not isinstance(tm, TransformManager):
        raise TypeError("Unkown transformation manager type '%s'"
                        % type(tm))

    urdf = BeautifulSoup(urdf_xml, "xml")
    robot = urdf.find("robot")
    if robot.has_attr("name"):
        robot_name = robot["name"]
    else:
        robot_name = "world"

    result = {}
    for link in robot.findAll("link"):
        name = link["name"]
        result[name] = Node(name, robot_name)

    for joint in robot.findAll("joint"):
        parent = joint.find("parent")
        parent_name = parent["link"]
        child = joint.find("child")
        child_name = child["link"]

        node = result[child_name]
        node.base = parent_name

        offset = np.fromstring(joint.find("origin")["xyz"], sep=" ")
        roll_pitch_yaw = np.fromstring(joint.find("origin")["rpy"], sep=" ")

        node.A2B = transform_from(matrix_from_rpy(roll_pitch_yaw), offset)

        if joint["type"] == "revolute":
            node.joint_name = joint["name"]
            node.joint_axis = np.fromstring(joint.find("axis")["xyz"], sep=" ")

    if tm is None:
        tm = TransformManagerWithJoints()

    for node in result.values():
        if isinstance(tm, TransformManagerWithJoints):
            tm.add_joint(node.joint_name, node.name, node.base, node.A2B,
                         node.joint_axis)
        else:
            tm.add_transform(node.name, node.base, node.A2B)

    return tm
