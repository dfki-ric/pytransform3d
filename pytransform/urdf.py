import numpy as np
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from, concat
from .rotations import matrix_from_axis_angle


class TransformManagerWithJoints(TransformManager):
    """Transformation manager that includes joints that can be moved."""
    def __init__(self):
        super(TransformManagerWithJoints, self).__init__()
        self.default_transformations = {}

    def add_joint(self, joint_name, from_frame, to_frame, A2B, axis):
        """Add joint.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        from_frame : string
            Child link of the joint

        to_frame : string
            Parent link of the joint

        A2B : array-like, shape (4, 4)
            Transformation from child to parent

        axis : array-like, shape (3,)
            Rotation axis of the joint (defined in the child frame)
        """
        self.add_transform(from_frame, to_frame, A2B)
        if joint_name is not None:
            self.default_transformations[joint_name] = (from_frame, to_frame,
                                                        A2B, axis)

    def set_joint(self, joint_name, angle):
        """Set joint angle.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        angle : float
            Joint angle in radians
        """
        from_frame, to_frame, A2B, axis = self.default_transformations[joint_name]
        joint_rotation = matrix_from_axis_angle(np.hstack((axis, [angle])))
        joint2A = transform_from(joint_rotation, np.zeros(3))
        self.add_transform(from_frame, to_frame, concat(joint2A, A2B))


class Node(object):
    """Node from URDF file."""
    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.A2B = np.eye(4)
        self.joint_name = None
        self.joint_axis = None


def _matrix_from_rpy(rpy):
    """Rotation matrix from fixed axis roll, pitch and yaw angles."""
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
    """Load URDF file into transformation manager.

    Parameters
    ----------
    urdf_xml : string
        Robot definition in URDF

    tm : TransformManager or TransformManagerWithJoints, optional
        Transformation manager

    Returns
    -------
    tm : TransformManager or TransformManagerWithJoints
        Returns a new transformation manager if none has been given. Otherwise
        the old transformation manager is returned.
    """
    if tm is not None and not isinstance(tm, TransformManager):
        raise TypeError("Unkown transformation manager type '%s'"
                        % type(tm))

    urdf = BeautifulSoup(urdf_xml, "xml")
    robot = urdf.find("robot")
    if robot.has_attr("name"):
        robot_name = robot["name"]
    else:
        robot_name = "world"

    nodes = {}
    for link in robot.findAll("link"):
        name = link["name"]
        nodes[name] = Node(name, robot_name)

    for joint in robot.findAll("joint"):
        parent = joint.find("parent")
        parent_name = parent["link"]
        child = joint.find("child")
        child_name = child["link"]

        node = nodes[child_name]
        node.base = parent_name

        offset = np.fromstring(joint.find("origin")["xyz"], sep=" ")
        roll_pitch_yaw = np.fromstring(joint.find("origin")["rpy"], sep=" ")

        node.A2B = transform_from(_matrix_from_rpy(roll_pitch_yaw), offset)

        if joint["type"] == "revolute":
            node.joint_name = joint["name"]
            node.joint_axis = np.fromstring(joint.find("axis")["xyz"], sep=" ")

    if tm is None:
        tm = TransformManagerWithJoints()

    for node in nodes.values():
        if isinstance(tm, TransformManagerWithJoints):
            tm.add_joint(node.joint_name, node.name, node.base, node.A2B,
                         node.joint_axis)
        else:
            tm.add_transform(node.name, node.base, node.A2B)

    return tm
