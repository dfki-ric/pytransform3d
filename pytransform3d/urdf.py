"""Load transformations from URDF files.

See :doc:`transform_manager` for more information.
"""
import os
import numpy as np
import warnings
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from, concat
from .rotations import (
    active_matrix_from_extrinsic_roll_pitch_yaw, matrix_from_axis_angle,
    norm_vector)


class UrdfTransformManager(TransformManager):
    """Transformation manager that can load URDF files.

    URDF is the `Unified Robot Description Format <http://wiki.ros.org/urdf>`_.
    URDF allows to define joints between links that can be rotated about one
    axis. This transformation manager allows to set the joint angles after
    joints have been added or loaded from an URDF.

    .. warning::

        Note that this module requires the Python package beautifulsoup4.

    .. note::

        Joint angles must be given in radians.

    Parameters
    ----------
    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrices are valid and requested nodes exist,
        which might significantly slow down some operations.
    """
    def __init__(self, strict_check=True, check=True):
        super(UrdfTransformManager, self).__init__(strict_check, check)
        self._joints = {}
        self.collision_objects = []
        self.visuals = []

    def add_joint(self, joint_name, from_frame, to_frame, child2parent, axis,
                  limits=(float("-inf"), float("inf")), joint_type="revolute"):
        """Add joint.

        Parameters
        ----------
        joint_name : str
            Name of the joint

        from_frame : Hashable
            Child link of the joint

        to_frame : Hashable
            Parent link of the joint

        child2parent : array-like, shape (4, 4)
            Transformation from child to parent

        axis : array-like, shape (3,)
            Rotation axis of the joint (defined in the child frame)

        limits : pair of float, optional (default: (-inf, inf))
            Lower and upper joint angle limit

        joint_type : str, optional (default: 'revolute')
            Joint type: revolute, prismatic, or fixed (continuous is the same
            as revolute)
        """
        self.add_transform(from_frame, to_frame, child2parent)
        self._joints[joint_name] = (
            from_frame, to_frame, child2parent, norm_vector(axis), limits,
            joint_type)

    def set_joint(self, joint_name, value):
        """Set joint position.

        Note that joint values are clipped to their limits.

        Parameters
        ----------
        joint_name : str
            Name of the joint

        value : float
            Joint angle in radians in case of revolute joints or position
            in case of prismatic joint.

        Raises
        ------
        KeyError
            If joint_name is unknown
        """
        if joint_name not in self._joints:
            raise KeyError("Joint '%s' is not known" % joint_name)
        from_frame, to_frame, child2parent, axis, limits, joint_type = \
            self._joints[joint_name]
        # this is way faster than np.clip:
        value = min(max(value, limits[0]), limits[1])
        if joint_type == "revolute":
            joint_rotation = matrix_from_axis_angle(
                np.hstack((axis, (value,))))
            joint2A = transform_from(
                joint_rotation, np.zeros(3), strict_check=self.strict_check)
        elif joint_type == "prismatic":
            joint_offset = value * axis
            joint2A = transform_from(
                np.eye(3), joint_offset, strict_check=self.strict_check)
        else:
            assert joint_type == "fixed"
            warnings.warn("Trying to set a fixed joint")
            return
        self.add_transform(from_frame, to_frame, concat(
            joint2A, child2parent, strict_check=self.strict_check,
            check=self.check))

    def get_joint_limits(self, joint_name):
        """Get limits of a joint.

        Parameters
        ----------
        joint_name : str
            Name of the joint

        Returns
        -------
        limits : pair of float
            Lower and upper joint angle limit

        Raises
        ------
        KeyError
            If joint_name is unknown
        """
        if joint_name not in self._joints:
            raise KeyError("Joint '%s' is not known" % joint_name)
        return self._joints[joint_name][4]

    def load_urdf(self, urdf_xml, mesh_path=None, package_dir=None):
        """Load URDF file into transformation manager.

        Parameters
        ----------
        urdf_xml : str
            Robot definition in URDF

        mesh_path : str, optional (default: None)
            Path in which we search for meshes that are defined in the URDF.
            Meshes will be ignored if it is set to None and no 'package_dir'
            is given.

        package_dir : str, optional (default: None)
            Some URDFs start file names with 'package://' to refer to the ROS
            package in which these files (textures, meshes) are located. This
            variable defines to which path this prefix will be resolved.
        """
        robot_name, links, joints = parse_urdf(
            urdf_xml, mesh_path, package_dir, self.strict_check)
        initialize_urdf_transform_manager(self, robot_name, links, joints)

    def plot_visuals(self, frame, ax=None, ax_s=1, wireframe=False,
                     convex_hull_of_mesh=True, alpha=0.3):  # pragma: no cover
        """Plot all visuals in a given reference frame.

        Visuals can be boxes, spheres, cylinders, or meshes. Note that visuals
        that cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        frame : Hashable
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: False)
            Plot wireframe (surface otherwise)

        convex_hull_of_mesh : bool, optional (default: True)
            Displays convex hull of meshes instead of the original mesh. This
            makes plotting a lot faster with complex meshes.

        alpha : float, optional (default: 0.3)
            Alpha value of the surface / wireframe that will be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        return self._plot_objects(
            self.visuals, frame, ax, ax_s, wireframe, convex_hull_of_mesh,
            alpha)

    def plot_collision_objects(
            self, frame, ax=None, ax_s=1, wireframe=True,
            convex_hull_of_mesh=True, alpha=1.0):  # pragma: no cover
        """Plot all collision objects in a given reference frame.

        Collision objects can be boxes, spheres, cylinders, or meshes. Note
        that collision objects that cannot be connected to the reference frame
        are omitted.

        Parameters
        ----------
        frame : Hashable
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe (surface otherwise)

        convex_hull_of_mesh : bool, optional (default: True)
            Displays convex hull of meshes instead of the original mesh. This
            makes plotting a lot faster with complex meshes.

        alpha : float, optional (default: 1)
            Alpha value of the surface / wireframe that will be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        return self._plot_objects(
            self.collision_objects, frame, ax, ax_s, wireframe,
            convex_hull_of_mesh, alpha)

    def _plot_objects(self, objects, frame, ax=None, ax_s=1, wireframe=True,
                      convex_hull_of_mesh=True, alpha=1.0):  # pragma: no cover
        """Plot all objects in a given reference frame.

        Objects can be boxes, spheres, cylinders, or meshes. Note that objects
        that cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        objects : list
            Objects that will be plotted

        frame : Hashable
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe (surface otherwise)

        convex_hull_of_mesh : bool, optional (default: True)
            Displays convex hull of meshes instead of the original mesh. This
            makes plotting a lot faster with complex meshes.

        alpha : float, optional (default: 1)
            Alpha value of the surface / wireframe that will be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            from .plot_utils import make_3d_axis
            ax = make_3d_axis(ax_s)
        for obj in objects:
            ax = obj.plot(
                self, frame, ax, wireframe=wireframe,
                convex_hull=convex_hull_of_mesh, alpha=alpha)
        return ax


def parse_urdf(urdf_xml, mesh_path=None, package_dir=None, strict_check=True):
    """Parse information from URDF file.

    Parameters
    ----------
    urdf_xml : str
        Robot definition in URDF

    mesh_path : str, optional (default: None)
        Path in which we search for meshes that are defined in the URDF.
        Meshes will be ignored if it is set to None and no 'package_dir'
        is given.

    package_dir : str, optional (default: None)
        Some URDFs start file names with 'package://' to refer to the ROS
        package in which these files (textures, meshes) are located. This
        variable defines to which path this prefix will be resolved.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    robot_name : str
        Name of the robot

    links : list of Link
        Links of the robot

    joints : list of Joint
        Joints of the robot

    Raises
    ------
    UrdfException
        If URDF is not valid
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

    materials = dict([
        _parse_material(material)
        for material in robot.findAll("material", recursive=False)])

    links = [_parse_link(link, materials, mesh_path, package_dir, strict_check)
             for link in robot.findAll("link", recursive=False)]

    link_names = [link.name for link in links]
    joints = [_parse_joint(joint, link_names, strict_check)
              for joint in robot.findAll("joint", recursive=False)]

    return robot_name, links, joints


def initialize_urdf_transform_manager(tm, robot_name, links, joints):
    """Initializes transform manager from previously parsed URDF data.

    Parameters
    ----------
    tm : UrdfTransformManager
        Transform manager

    robot_name : str
        Name of the robot

    links : list of Link
        Links of the robot

    joints : list of Joint
        Joints of the robot
    """
    tm.add_transform(links[0].name, robot_name, np.eye(4))
    _add_links(tm, links)
    _add_joints(tm, joints)


def _parse_material(material):
    """Parse material."""
    if not material.has_attr("name"):
        raise UrdfException("Material name is missing.")
    colors = material.findAll("color")
    if len(colors) not in [0, 1]:
        raise UrdfException("More than one color is not allowed.")
    if len(colors) == 1:
        color = _parse_color(colors[0])
    else:
        color = None
    # TODO texture is currently ignored
    return material["name"], color


def _parse_color(color):
    """Parse color."""
    if not color.has_attr("rgba"):
        raise UrdfException("Attribute 'rgba' of color tag is missing.")
    return np.fromstring(color["rgba"], sep=" ")


def _parse_link(link, materials, mesh_path, package_dir, strict_check):
    """Create link."""
    if not link.has_attr("name"):
        raise UrdfException("Link name is missing.")

    result = Link()
    result.name = link["name"]

    visuals, visual_transforms = _parse_link_children(
        link, "visual", materials, mesh_path, package_dir, strict_check)
    result.visuals = visuals
    result.transforms.extend(visual_transforms)

    collision_objects, collision_object_transforms = _parse_link_children(
        link, "collision", dict(), mesh_path, package_dir, strict_check)
    result.collision_objects = collision_objects
    result.transforms.extend(collision_object_transforms)

    inertial = link.find("inertial")
    if inertial is not None:
        result.inertial_frame[:, :] = _parse_origin(inertial, strict_check)
        result.mass = _parse_mass(inertial)
        result.inertia[:, :] = _parse_inertia(inertial)
        result.transforms.append(
            ("inertial_frame:%s" % result.name, result.name,
             result.inertial_frame))

    return result


def _parse_link_children(link, child_type, materials, mesh_path, package_dir,
                         strict_check):
    """Parse collision objects or visuals."""
    children = link.findAll(child_type)
    shape_objects = []
    transforms = []
    for i, child in enumerate(children):
        if child.has_attr("name"):
            name = "%s:%s/%s" % (child_type, link["name"], child["name"])
        else:
            name = "%s:%s/%s" % (child_type, link["name"], i)

        color = None
        if child_type == "visual":
            material = child.find("material")
            if material is not None:
                material_name, color = _parse_material(material)
                if color is None and material_name in materials:
                    color = materials[material_name]

        child2link = _parse_origin(child, strict_check)
        transforms.append((name, link["name"], child2link))

        shape_objects.extend(_parse_geometry(
            child, name, color, mesh_path, package_dir))
    return shape_objects, transforms


def _parse_geometry(child, name, color, mesh_path, package_dir):
    """Parse geometric primitives (box, cylinder, sphere) or meshes."""
    geometry = child.find("geometry")
    if geometry is None:
        raise UrdfException("Missing geometry tag in link '%s'" % name)
    result = []
    for shape_type in ["box", "cylinder", "sphere", "mesh"]:
        shapes = geometry.findAll(shape_type)
        Cls = shape_classes[shape_type]
        for shape in shapes:
            shape_object = Cls(
                name, mesh_path=mesh_path, package_dir=package_dir,
                color=color)
            shape_object.parse(shape)
            result.append(shape_object)
    return result


def _parse_origin(entry, strict_check):
    """Parse transformation."""
    origin = entry.find("origin")
    translation = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        if origin.has_attr("xyz"):
            translation = np.fromstring(origin["xyz"], sep=" ")
        if origin.has_attr("rpy"):
            roll_pitch_yaw = np.fromstring(origin["rpy"], sep=" ")
            # URDF and KDL use the active convention for rotation matrices.
            # For more details on how the URDF parser handles the
            # conversion from Euler angles, see this blog post:
            # https://orbitalstation.wordpress.com/tag/quaternion/
            rotation = active_matrix_from_extrinsic_roll_pitch_yaw(
                roll_pitch_yaw)
    return transform_from(
        rotation, translation, strict_check=strict_check)


def _parse_mass(inertial):
    """Parse link mass."""
    mass = inertial.find("mass")
    if mass is not None and mass.has_attr("value"):
        result = float(mass["value"])
    else:
        result = 0.0
    return result


def _parse_inertia(inertial):
    """Parse inertia matrix."""
    inertia = inertial.find("inertia")

    result = np.zeros((3, 3))
    if inertia is None:
        return result

    if inertia.has_attr("ixx"):
        result[0, 0] = float(inertia["ixx"])
    if inertia.has_attr("ixy"):
        ixy = float(inertia["ixy"])
        result[0, 1] = ixy
        result[1, 0] = ixy
    if inertia.has_attr("ixz"):
        ixz = float(inertia["ixz"])
        result[0, 2] = ixz
        result[2, 0] = ixz
    if inertia.has_attr("iyy"):
        result[1, 1] = float(inertia["iyy"])
    if inertia.has_attr("iyz"):
        iyz = float(inertia["iyz"])
        result[1, 2] = iyz
        result[2, 1] = iyz
    if inertia.has_attr("izz"):
        result[2, 2] = float(inertia["izz"])
    return result


def _parse_joint(joint, link_names, strict_check):
    """Create joint object."""
    j = Joint()

    if not joint.has_attr("name"):
        raise UrdfException("Joint name is missing.")
    j.joint_name = joint["name"]

    if not joint.has_attr("type"):
        raise UrdfException("Joint type is missing in joint '%s'."
                            % j.joint_name)

    parent = joint.find("parent")
    if parent is None:
        raise UrdfException("No parent specified in joint '%s'"
                            % j.joint_name)
    if not parent.has_attr("link"):
        raise UrdfException("No parent link name given in joint '%s'."
                            % j.joint_name)
    j.parent = parent["link"]
    if j.parent not in link_names:
        raise UrdfException("Parent link '%s' of joint '%s' is not "
                            "defined." % (j.parent, j.joint_name))

    child = joint.find("child")
    if child is None:
        raise UrdfException("No child specified in joint '%s'"
                            % j.joint_name)
    if not child.has_attr("link"):
        raise UrdfException("No child link name given in joint '%s'."
                            % j.joint_name)
    j.child = child["link"]
    if j.child not in link_names:
        raise UrdfException("Child link '%s' of joint '%s' is not "
                            "defined." % (j.child, j.joint_name))

    j.joint_type = joint["type"]

    if j.joint_type in ["planar", "floating"]:
        raise UrdfException("Unsupported joint type '%s'" % j.joint_type)
    if j.joint_type not in ["revolute", "continuous", "prismatic",
                            "fixed"]:
        raise UrdfException("Joint type '%s' is not allowed in a URDF "
                            "document." % j.joint_type)

    j.child2parent = _parse_origin(joint, strict_check)

    j.joint_axis = np.array([1, 0, 0])
    if j.joint_type in ["revolute", "continuous", "prismatic"]:
        axis = joint.find("axis")
        if axis is not None and axis.has_attr("xyz"):
            j.joint_axis = np.fromstring(axis["xyz"], sep=" ")

    j.limits = _parse_limits(joint)
    return j


def _parse_limits(joint):
    """Parse joint limits."""
    limit = joint.find("limit")
    lower, upper = float("-inf"), float("inf")
    if limit is not None:
        if limit.has_attr("lower"):
            lower = float(limit["lower"])
        if limit.has_attr("upper"):
            upper = float(limit["upper"])
    return lower, upper


def _add_links(tm, links):
    """Add previously parsed links.

    Parameters
    ----------
    tm : UrdfTransformManager
        Transform manager

    links : list of Link
        Joint information from URDF
    """
    for link in links:
        tm.visuals.extend(link.visuals)
        tm.collision_objects.extend(link.collision_objects)

        for from_frame, to_frame, transform in link.transforms:
            tm.add_transform(from_frame, to_frame, transform)


def _add_joints(tm, joints):
    """Add previously parsed joints.

    Parameters
    ----------
    tm : UrdfTransformManager
        Transform manager

    joints : list of Joint
        Joint information from URDF
    """
    for joint in joints:
        if joint.joint_type in ["revolute", "continuous"]:
            tm.add_joint(
                joint.joint_name, joint.child, joint.parent,
                joint.child2parent, joint.joint_axis, joint.limits,
                "revolute")
        elif joint.joint_type == "prismatic":
            tm.add_joint(
                joint.joint_name, joint.child, joint.parent,
                joint.child2parent, joint.joint_axis, joint.limits,
                "prismatic")
        else:
            assert joint.joint_type == "fixed"
            tm.add_joint(
                joint.joint_name, joint.child, joint.parent,
                joint.child2parent, joint.joint_axis, (0.0, 0.0),
                "fixed")


class Link(object):
    """Link from URDF file.

    This class is only required temporarily while we parse the URDF.

    Attributes
    ----------
    name : str
        Link name

    visuals : list of Geometry
        Visual geometries

    collision_objects : list of Geometry
        Geometries for collision calculation

    transforms : list
        Transformations given as tuples: name of frame A, name of frame B,
        transform A2B

    inertial_frame : array, shape (4, 4)
        Pose of inertial frame with respect to the link

    mass : float
        Mass of the link

    inertia : array, shape (3, 3)
        Inertia matrix
    """
    def __init__(self):
        self.name = None
        self.visuals = []
        self.collision_objects = []
        self.transforms = []
        self.inertial_frame = np.eye(4)
        self.mass = 0.0
        self.inertia = np.zeros((3, 3))


class Joint(object):
    """Joint from URDF file.

    This class is only required temporarily while we parse the URDF.

    Attributes
    ----------
    child : str
        Name of the child

    parent : str
        Name of the parent frame

    child2parent : array-like, shape (4, 4)
        Transformation from child to parent

    joint_name : str
        Name of the joint that defines the transformation

    joint_axis : array-like, shape (3,)
        Rotation axis of the joint (defined in the child frame)

    joint_type : str
        Either 'fixed' or 'revolute'

    limits : pair of float
        Lower and upper joint angle limit
    """
    def __init__(self):
        self.child = None
        self.parent = None
        self.child2parent = np.eye(4)
        self.joint_name = None
        self.joint_axis = None
        self.joint_type = "fixed"
        self.limits = float("-inf"), float("inf")


class Geometry(object):
    """Geometrical object."""
    def __init__(self, frame, mesh_path, package_dir, color):
        self.frame = frame
        self.mesh_path = mesh_path
        self.package_dir = package_dir
        self.color = color

    def parse(self, xml):
        """Parse parameters of geometry."""

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True,
             convex_hull=True):
        """Plot geometry."""


class Box(Geometry):
    """Geometrical object: box."""
    def __init__(self, frame, mesh_path, package_dir, color):
        super(Box, self).__init__(frame, mesh_path, package_dir, color)
        self.size = np.zeros(3)

    def parse(self, xml):
        """Parse box size."""
        if xml.has_attr("size"):
            self.size[:] = np.fromstring(xml["size"], sep=" ")

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True,
             convex_hull=True):  # pragma: no cover
        """Plot box."""
        A2B = tm.get_transform(self.frame, frame)
        color = self.color if self.color is not None else "k"
        from .plot_utils import plot_box
        return plot_box(
            ax, self.size, A2B, wireframe=wireframe, alpha=alpha, color=color)


class Sphere(Geometry):
    """Geometrical object: sphere."""
    def __init__(self, frame, mesh_path, package_dir, color):
        super(Sphere, self).__init__(frame, mesh_path, package_dir, color)
        self.radius = 0.0

    def parse(self, xml):
        """Parse sphere radius."""
        if not xml.has_attr("radius"):
            raise UrdfException("Sphere has no radius.")
        self.radius = float(xml["radius"])

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True,
             convex_hull=True):  # pragma: no cover
        """Plot sphere."""
        center = tm.get_transform(self.frame, frame)[:3, 3]
        color = self.color if self.color is not None else "k"
        from .plot_utils import plot_sphere
        return plot_sphere(
            ax, self.radius, center, wireframe=wireframe, alpha=alpha,
            color=color)


class Cylinder(Geometry):
    """Geometrical object: cylinder."""
    def __init__(self, frame, mesh_path, package_dir, color):
        super(Cylinder, self).__init__(frame, mesh_path, package_dir, color)
        self.radius = 0.0
        self.length = 0.0

    def parse(self, xml):
        """Parse cylinder radius and length."""
        if not xml.has_attr("radius"):
            raise UrdfException("Cylinder has no radius.")
        self.radius = float(xml["radius"])
        if not xml.has_attr("length"):
            raise UrdfException("Cylinder has no length.")
        self.length = float(xml["length"])

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True,
             convex_hull=True):  # pragma: no cover
        """Plot cylinder."""
        A2B = tm.get_transform(self.frame, frame)
        color = self.color if self.color is not None else "k"
        from .plot_utils import plot_cylinder
        return plot_cylinder(
            ax, self.length, self.radius, 0.0, A2B, wireframe=wireframe,
            alpha=alpha, color=color)


class Mesh(Geometry):
    """Geometrical object: mesh."""
    def __init__(self, frame, mesh_path, package_dir, color):
        super(Mesh, self).__init__(frame, mesh_path, package_dir, color)
        self.filename = None
        self.scale = np.ones(3)

    def parse(self, xml):
        """Parse mesh filename and scale."""
        if self.mesh_path is None and self.package_dir is None:
            self.filename = None
        else:
            if not xml.has_attr("filename"):
                raise UrdfException("Mesh has no filename.")
            if self.mesh_path is not None:
                self.filename = os.path.join(self.mesh_path, xml["filename"])
            else:
                assert self.package_dir is not None
                self.filename = xml["filename"].replace(
                    "package://", self.package_dir)
            if xml.has_attr("scale"):
                self.scale = np.fromstring(xml["scale"], sep=" ")

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True,
             convex_hull=True):  # pragma: no cover
        """Plot mesh."""
        from .plot_utils import plot_mesh
        A2B = tm.get_transform(self.frame, frame)
        color = self.color if self.color is not None else "k"
        return plot_mesh(
            ax, self.filename, A2B, self.scale, wireframe=wireframe,
            convex_hull=convex_hull, alpha=alpha, color=color)


shape_classes = {"box": Box,
                 "sphere": Sphere,
                 "cylinder": Cylinder,
                 "mesh": Mesh}


class UrdfException(Exception):
    """Exception while parsing URDF files."""
