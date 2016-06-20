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
        self.default_transformations[joint_name] = (from_frame, to_frame, A2B,
                                                    axis)

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
        self.joint_type = "fixed"


class UrdfException(Exception):
    pass


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
    if robot is None:
        raise UrdfException("Robot tag is missing.")

    if not robot.has_attr("name"):
        raise UrdfException("Attribute 'name' is missing in robot tag.")

    robot_name = robot["name"]

    nodes = {}
    for link in robot.findAll("link"):
        node = _parse_link(link, nodes, robot_name)
        nodes[node.name] = node

    for joint in robot.findAll("joint"):
        if not joint.has_attr("name"):
            raise UrdfException("Joint name is missing.")
        joint_name = joint["name"]

        if not joint.has_attr("type"):
            raise UrdfException("Joint type is missing in joint '%s'."
                                % joint_name)

        parent = joint.find("parent")
        if parent is None:
            raise UrdfException("No parent specified in joint '%s'"
                                % joint_name)

        child = joint.find("child")
        if child is None:
            raise UrdfException("No child specified in joint '%s'"
                                % joint_name)

        if not parent.has_attr("link"):
            raise UrdfException("No parent link name given in joint '%s'."
                                % joint_name)
        parent_name = parent["link"]

        if not child.has_attr("link"):
            raise UrdfException("No child link name given in joint '%s'."
                                % joint_name)
        child_name = child["link"]

        if child_name not in nodes:
            raise UrdfException("Child link '%s' of joint '%s' is not defined."
                                % (child_name, joint_name))
        node = nodes[child_name]
        node.joint_name = joint_name
        node.joint_type = joint["type"]

        if parent_name not in nodes:
            raise UrdfException("Parent link '%s' of joint '%s' is not defined."
                                % (parent_name, joint_name))
        node.base = parent_name

        origin = joint.find("origin")
        translation = np.zeros(3)
        rotation = np.eye(3)
        if origin is not None:
            if origin.has_attr("xyz"):
                translation = np.fromstring(origin["xyz"], sep=" ")
            if origin.has_attr("rpy"):
                roll_pitch_yaw = np.fromstring(origin["rpy"], sep=" ")
                rotation = _matrix_from_rpy(roll_pitch_yaw)

        node.A2B = transform_from(rotation, translation)

        node.joint_axis = np.array([1, 0, 0])
        if node.joint_type == "revolute":
            axis = joint.find("axis")
            if axis is not None and axis.has_attr("xyz"):
                node.joint_axis = np.fromstring(axis["xyz"], sep=" ")

    if tm is None:
        tm = TransformManagerWithJoints()

    for node in nodes.values():
        if (isinstance(tm, TransformManagerWithJoints) and
                    node.joint_type == "revolute"):
            tm.add_joint(node.joint_name, node.name, node.base, node.A2B,
                         node.joint_axis)
        else:
            tm.add_transform(node.name, node.base, node.A2B)

    return tm


def _parse_link(link, nodes, robot_name):
    if not link.has_attr("name"):
        raise UrdfException("Link name is missing.")
    return Node(link["name"], robot_name)
