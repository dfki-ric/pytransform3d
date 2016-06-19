import numpy as np
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from


class Node(object):
    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.A2B = np.eye(4)


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

    if tm is None:
        tm = TransformManager()

    for node in result.values():
        tm.add_transform(node.name, node.base, node.A2B)

    return tm
