import numpy as np
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from, concat, transform
from .rotations import matrix_from_euler_xyz, matrix_from_axis_angle
from .plot_utils import make_3d_axis


class UrdfTransformManager(TransformManager):
    """Transformation manager that can load URDF files.

    URDF is the `Unified Robot Description Format <http://wiki.ros.org/urdf>`_.
    URDF allows to define joints between links that can be rotated about one
    axis. This transformation manager allows to set the joint angles after
    joints have been added or loaded from an URDF.

    .. warning::

        Note that this module requires the Python package beautifulsoup4.
    """
    def __init__(self):
        super(UrdfTransformManager, self).__init__()
        self._joints = {}
        self.collision_objects = []
        self.visuals = []

    def add_joint(self, joint_name, from_frame, to_frame, child2parent, axis):
        """Add joint.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        from_frame : string
            Child link of the joint

        to_frame : string
            Parent link of the joint

        child2parent : array-like, shape (4, 4)
            Transformation from child to parent

        axis : array-like, shape (3,)
            Rotation axis of the joint (defined in the child frame)
        """
        self.add_transform(from_frame, to_frame, child2parent)
        self._joints[joint_name] = from_frame, to_frame, child2parent, axis

    def set_joint(self, joint_name, angle):
        """Set joint angle.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        angle : float
            Joint angle in radians
        """
        if joint_name not in self._joints:
            raise KeyError("Joint '%s' is not known" % joint_name)
        from_frame, to_frame, child2parent, axis = self._joints[joint_name]
        joint_rotation = matrix_from_axis_angle(np.hstack((axis, [angle])))
        joint2A = transform_from(joint_rotation, np.zeros(3))
        self.add_transform(from_frame, to_frame, concat(joint2A, child2parent))

    def load_urdf(self, urdf_xml):
        """Load URDF file into transformation manager.

        Parameters
        ----------
        urdf_xml : string
            Robot definition in URDF
        """
        urdf = BeautifulSoup(urdf_xml, "xml")

        # URDF XML schema:
        # https://github.com/ros/urdfdom/blob/master/xsd/urdf.xsd

        robot = urdf.find("robot")
        if robot is None:
            raise UrdfException("Robot tag is missing.")

        if not robot.has_attr("name"):
            raise UrdfException("Attribute 'name' is missing in robot tag.")

        robot_name = robot["name"]

        nodes = {}
        for link in robot.findAll("link"):
            node = self._parse_link(link, robot_name)
            nodes[node.child] = node

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
            if not parent.has_attr("link"):
                raise UrdfException("No parent link name given in joint '%s'."
                                    % joint_name)
            parent_name = parent["link"]
            if parent_name not in nodes:
                raise UrdfException("Parent link '%s' of joint '%s' is not "
                                    "defined." % (parent_name, joint_name))

            child = joint.find("child")
            if child is None:
                raise UrdfException("No child specified in joint '%s'"
                                    % joint_name)
            if not child.has_attr("link"):
                raise UrdfException("No child link name given in joint '%s'."
                                    % joint_name)
            child_name = child["link"]
            if child_name not in nodes:
                raise UrdfException("Child link '%s' of joint '%s' is not "
                                    "defined." % (child_name, joint_name))

            self._parse_joint(joint, nodes[child_name], parent_name)

        for node in nodes.values():
            if node.joint_type == "revolute":
                self.add_joint(node.joint_name, node.child, node.parent,
                               node.child2parent, node.joint_axis)
            else:
                self.add_transform(node.child, node.parent, node.child2parent)

    def _parse_link(self, link, robot_name):
        """Make node from link."""
        if not link.has_attr("name"):
            raise UrdfException("Link name is missing.")
        self.visuals = self._parse_link_children(link, "visual")
        self.collision_objects = self._parse_link_children(link, "collision")
        return Node(link["name"], robot_name)

    def _parse_link_children(self, link, child_type):
        """Parse collision objects or visuals."""
        children = link.findAll(child_type)
        shape_objects = []
        for i, child in enumerate(children):
            if child.has_attr("name"):
                name = child["name"]
            else:
                name = "%s/%s_%s" % (link["name"], child_type, i)
            child2link = self._parse_origin(child)
            self.add_transform(name, link["name"], child2link)
            shape_objects.extend(self._parse_geometry(child, name))
        return shape_objects

    def _parse_geometry(self, child, name):
        """Parse geometric primitives (box, cylinder, sphere).

        Meshes are not supported!
        """
        geometry = child.find("geometry")
        if geometry is None:
            raise UrdfException("Missing geometry tag in link '%s'" % name)
        result = []
        # meshes are not supported
        for shape_type in ["box", "cylinder", "sphere"]:
            shapes = geometry.findAll(shape_type)
            Cls = shape_classes[shape_type]
            for shape in shapes:
                shape_object = Cls(name)
                shape_object.parse(shape)
                result.append(shape_object)
        return result

    def _parse_joint(self, joint, node, parent_name):
        """Update node with joint."""
        node.joint_name = joint["name"]
        node.joint_type = joint["type"]
        node.parent = parent_name

        if node.joint_type in ["planar", "floating", "continuous",
                               "prismatic"]:
            raise UrdfException("Unsupported joint type '%s'"
                                % node.joint_type)
        elif node.joint_type not in ["revolute", "fixed"]:
            raise UrdfException("Joint type '%s' is not allowed in a URDF "
                                "document." % node.joint_type)

        node.child2parent = self._parse_origin(joint)

        node.joint_axis = np.array([1, 0, 0])
        if node.joint_type == "revolute":
            axis = joint.find("axis")
            if axis is not None and axis.has_attr("xyz"):
                node.joint_axis = np.fromstring(axis["xyz"], sep=" ")

    def _parse_origin(self, entry):
        """Parse transformation."""
        origin = entry.find("origin")
        translation = np.zeros(3)
        rotation = np.eye(3)
        if origin is not None:
            if origin.has_attr("xyz"):
                translation = np.fromstring(origin["xyz"], sep=" ")
            if origin.has_attr("rpy"):
                roll_pitch_yaw = np.fromstring(origin["rpy"], sep=" ")
                # URDF and KDL use the alias convention for rotation matrices
                # instead of alibi convention. That means the reference frame
                # is rotated by the rotation matrix and not the point. To
                # convert the defined rotation to the alibi convention we must
                # invert (transpose) the matrix.
                # For more details on how the URDF parser handles the
                # conversion from Euler angles, see this blog post:
                # https://orbitalstation.wordpress.com/tag/quaternion/
                rotation = matrix_from_euler_xyz(roll_pitch_yaw).T
        return transform_from(rotation, translation)

    def plot_visuals(self, frame, ax=None, ax_s=1):
        """Plot all visuals in a given reference frame.

        Visuals can be boxes, spheres, or cylinders. Note that visuals that
        cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        frame : string
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        return self._plot_objects(self.visuals, frame, ax, ax_s)

    def plot_collision_objects(self, frame, ax=None, ax_s=1):
        """Plot all collision objects in a given reference frame.

        Collision objects can be boxes, spheres, or cylinders. Note that
        collision objects that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : string
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        return self._plot_objects(self.collision_objects, frame, ax, ax_s)

    def _plot_objects(self, objects, frame, ax=None, ax_s=1):
        """Plot all objects in a given reference frame.

        Objects can be boxes, spheres, or cylinders. Note that objects that
        cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        frame : string
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)
        for obj in objects:
            ax = obj.plot(self, frame, ax)
        return ax


class Node(object):
    """Node from URDF file.

    This class is only required temporarily while we parse the URDF.

    Parameters
    ----------
    child : string
        Name of the child

    parent : string
        Name of the parent frame

    Attributes
    ----------
    child : string
        Name of the child

    parent : string
        Name of the parent frame

    child2parent : array-like, shape (4, 4)
        Transformation from child to parent

    joint_name : string
        Name of the joint that defines the transformation

    joint_axis : array-like, shape (3,)
        Rotation axis of the joint (defined in the child frame)

    joint_type : string
        Either 'fixed' or 'revolute'
    """
    def __init__(self, child, parent):
        self.child = child
        self.parent = parent
        self.child2parent = np.eye(4)
        self.joint_name = None
        self.joint_axis = None
        self.joint_type = "fixed"


class Box(object):
    def __init__(self, frame):
        self.frame = frame

    def parse(self, box):
        self.size = np.zeros(3)
        if box.has_attr("size"):
            self.size = np.fromstring(box["size"], sep=" ")

    def plot(self, tm, frame, ax=None, color="k"):
        A2B = tm.get_transform(self.frame, frame)

        corners = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ]) * self.size
        corners = transform(
            A2B, np.hstack((corners, np.ones((len(corners), 1)))))[:, :3]

        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3),
                     (4, 5), (4, 6), (5, 7), (6, 7),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]], c=color)

        return ax


class Sphere(object):
    def __init__(self, frame):
        self.frame = frame

    def parse(self, sphere):
        if not sphere.has_attr("radius"):
            raise UrdfException("Sphere has no radius.")
        self.radius = float(sphere["radius"])

    def plot(self, tm, frame, ax=None, color="k"):
        center = tm.get_transform(self.frame, frame)[:3, 3]
        phi, theta = np.mgrid[0.0:np.pi:30j, 0.0:2.0 * np.pi:30j]
        x = center[0] + self.radius * np.sin(phi) * np.cos(theta)
        y = center[1] + self.radius * np.sin(phi) * np.sin(theta)
        z = center[2] + self.radius * np.cos(phi)
        ax.plot_surface(x, y, z, rstride=5, cstride=5, color=color,
                        alpha=0.1, linewidth=0)
        ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color=color,
                          alpha=0.1)

        return ax


class Cylinder(object):
    def __init__(self, frame):
        self.frame = frame

    def parse(self, cylinder):
        if not cylinder.has_attr("radius"):
            raise UrdfException("Sphere has no radius.")
        self.radius = float(cylinder["radius"])
        if not cylinder.has_attr("length"):
            raise UrdfException("Sphere has no length.")
        self.length = float(cylinder["length"])

    def plot(self, tm, frame, ax=None, color="k"):
        A2B = tm.get_transform(self.frame, frame)

        axis_start = A2B.dot(np.array([0, 0, -0.5 * self.length, 1]))[:3]
        axis_end = A2B.dot(np.array([0, 0, 0.5 * self.length, 1]))[:3]
        axis = axis_end - axis_start
        axis /= self.length

        not_axis = np.array([1, 0, 0])
        if (axis == not_axis).all():
            not_axis = np.array([0, 1, 0])

        n1 = np.cross(axis, not_axis)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(axis, n1)

        t = np.linspace(0, self.length, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        t, theta = np.meshgrid(t, theta)
        X, Y, Z = [axis_start[i] + axis[i] * t +
                   self.radius * np.sin(theta) * n1[i] +
                   self.radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        ax.plot_surface(X, Y, Z, alpha=0.2, color=color)

        return ax


shape_classes = {"box": Box,
                 "sphere": Sphere,
                 "cylinder": Cylinder}


class UrdfException(Exception):
    """Exception while parsing URDF files."""
    pass
