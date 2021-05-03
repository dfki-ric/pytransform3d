"""Load transformations from URDF files.

See :doc:`transform_manager` for more information.
"""
import os
import numpy as np
from bs4 import BeautifulSoup
from .transform_manager import TransformManager
from .transformations import transform_from, concat
from .rotations import (
    active_matrix_from_extrinsic_roll_pitch_yaw, matrix_from_axis_angle,
    norm_vector)
from .plot_utils import (
    make_3d_axis, plot_mesh, plot_cylinder, plot_sphere, plot_box)


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
        self.mesh_path = None
        self.package_dir = None

    def add_joint(self, joint_name, from_frame, to_frame, child2parent, axis,
                  limits=(float("-inf"), float("inf")), joint_type="revolute"):
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

        limits : pair of float, optional (default: (-inf, inf))
            Lower and upper joint angle limit

        joint_type : str, optional (default: 'revolute')
            Joint type: revolute or prismatic (continuous is the same as
            revolute)
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
        joint_name : string
            Name of the joint

        value : float
            Joint angle in radians in case of revolute joints or position
            in case of prismatic joint.
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
        else:
            assert joint_type == "prismatic"
            joint_offset = value * axis
            joint2A = transform_from(
                np.eye(3), joint_offset, strict_check=self.strict_check)
        self.add_transform(from_frame, to_frame, concat(
            joint2A, child2parent, strict_check=self.strict_check,
            check=self.check))

    def get_joint_limits(self, joint_name):
        """Get limits of a joint.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        Returns
        -------
        limits : pair of float
            Lower and upper joint angle limit
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
        self.mesh_path = mesh_path
        self.package_dir = package_dir
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
            self._parse_material(material)
            for material in robot.findAll("material", recursive=False)])

        links = [self._parse_link(link, materials)
                 for link in robot.findAll("link", recursive=False)]
        joints = [self._parse_joint(joint, links)
                  for joint in robot.findAll("joint", recursive=False)]

        self.add_transform(links[0], robot_name, np.eye(4))
        for joint in joints:
            if joint.joint_type in ["revolute", "continuous"]:
                self.add_joint(
                    joint.joint_name, joint.child, joint.parent,
                    joint.child2parent, joint.joint_axis, joint.limits,
                    "revolute")
            elif joint.joint_type == "prismatic":
                self.add_joint(
                    joint.joint_name, joint.child, joint.parent,
                    joint.child2parent, joint.joint_axis, joint.limits,
                    "prismatic")
            else:
                assert joint.joint_type == "fixed"
                self.add_transform(
                    joint.child, joint.parent, joint.child2parent)

    def _parse_material(self, material):
        """Parse material."""
        if not material.has_attr("name"):
            raise UrdfException("Material name is missing.")
        colors = material.findAll("color")
        if len(colors) not in [0, 1]:
            raise UrdfException("More than one color is not allowed.")
        if len(colors) == 1:
            color = self._parse_color(colors[0])
        else:
            color = None
        # TODO texture is currently ignored
        return material["name"], color

    def _parse_color(self, color):
        """Parse color."""
        if not color.has_attr("rgba"):
            raise UrdfException("Attribute 'rgba' of color tag is missing.")
        return np.fromstring(color["rgba"], sep=" ")

    def _parse_link(self, link, materials):
        """Create link."""
        if not link.has_attr("name"):
            raise UrdfException("Link name is missing.")
        self.visuals.extend(
            self._parse_link_children(link, "visual", materials))
        self.collision_objects.extend(
            self._parse_link_children(link, "collision", dict()))
        return link["name"]

    def _parse_link_children(self, link, child_type, materials):
        """Parse collision objects or visuals."""
        children = link.findAll(child_type)
        shape_objects = []
        for i, child in enumerate(children):
            if child.has_attr("name"):
                name = "%s:%s/%s" % (child_type, link["name"], child["name"])
            else:
                name = "%s:%s/%s" % (child_type, link["name"], i)

            color = None
            if child_type == "visual":
                material = child.find("material")
                if material is not None:
                    material_name, color = self._parse_material(material)
                    if color is None and material_name in materials:
                        color = materials[material_name]

            child2link = self._parse_origin(child)
            self.add_transform(name, link["name"], child2link)

            shape_objects.extend(self._parse_geometry(child, name, color))
        return shape_objects

    def _parse_geometry(self, child, name, color):
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
                    name, mesh_path=self.mesh_path,
                    package_dir=self.package_dir, color=color)
                shape_object.parse(shape)
                result.append(shape_object)
        return result

    def _parse_joint(self, joint, links):
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
        if j.parent not in links:
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
        if j.child not in links:
            raise UrdfException("Child link '%s' of joint '%s' is not "
                                "defined." % (j.child, j.joint_name))

        j.joint_type = joint["type"]

        if j.joint_type in ["planar", "floating"]:
            raise UrdfException("Unsupported joint type '%s'" % j.joint_type)
        if j.joint_type not in ["revolute", "continuous", "prismatic",
                                "fixed"]:
            raise UrdfException("Joint type '%s' is not allowed in a URDF "
                                "document." % j.joint_type)

        j.child2parent = self._parse_origin(joint)

        j.joint_axis = np.array([1, 0, 0])
        if j.joint_type in ["revolute", "continuous", "prismatic"]:
            axis = joint.find("axis")
            if axis is not None and axis.has_attr("xyz"):
                j.joint_axis = np.fromstring(axis["xyz"], sep=" ")

        j.limits = self._parse_limits(joint)
        return j

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
                # URDF and KDL use the active convention for rotation matrices.
                # For more details on how the URDF parser handles the
                # conversion from Euler angles, see this blog post:
                # https://orbitalstation.wordpress.com/tag/quaternion/
                rotation = active_matrix_from_extrinsic_roll_pitch_yaw(
                    roll_pitch_yaw)
        return transform_from(
            rotation, translation, strict_check=self.strict_check)

    def _parse_limits(self, joint):
        """Parse joint limits."""
        limit = joint.find("limit")
        lower, upper = float("-inf"), float("inf")
        if limit is not None:
            if limit.has_attr("lower"):
                lower = float(limit["lower"])
            if limit.has_attr("upper"):
                upper = float(limit["upper"])
        return lower, upper

    def plot_visuals(self, frame, ax=None, ax_s=1, wireframe=False, convex_hull_of_mesh=True, alpha=0.3):
        """Plot all visuals in a given reference frame.

        Visuals can be boxes, spheres, cylinders, or meshes. Note that visuals
        that cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        frame : string
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

    def plot_collision_objects(self, frame, ax=None, ax_s=1, wireframe=True, convex_hull_of_mesh=True, alpha=1.0):
        """Plot all collision objects in a given reference frame.

        Collision objects can be boxes, spheres, cylinders, or meshes. Note
        that collision objects that cannot be connected to the reference frame
        are omitted.

        Parameters
        ----------
        frame : string
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

    def _plot_objects(self, objects, frame, ax=None, ax_s=1, wireframe=True, convex_hull_of_mesh=True, alpha=1.0):
        """Plot all objects in a given reference frame.

        Objects can be boxes, spheres, cylinders, or meshes. Note that objects
        that cannot be connected to the reference frame are omitted.

        Parameters
        ----------
        objects : list
            Objects that will be plotted

        frame : string
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
            ax = make_3d_axis(ax_s)
        for obj in objects:
            ax = obj.plot(
                self, frame, ax, wireframe=wireframe,
                convex_hull=convex_hull_of_mesh, alpha=alpha)
        return ax


class Joint(object):
    """Joint from URDF file.

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


class Box(object):
    def __init__(self, frame, mesh_path, package_dir, color):
        self.frame = frame
        self.color = color
        self.size = np.zeros(3)

    def parse(self, box):
        if box.has_attr("size"):
            self.size[:] = np.fromstring(box["size"], sep=" ")

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True, convex_hull=True):
        A2B = tm.get_transform(self.frame, frame)
        color = self.color if self.color is not None else "k"
        return plot_box(
            ax, self.size, A2B, wireframe=wireframe, alpha=alpha, color=color)


class Sphere(object):
    def __init__(self, frame, mesh_path, package_dir, color):
        self.frame = frame
        self.color = color
        self.radius = 0.0

    def parse(self, sphere):
        if not sphere.has_attr("radius"):
            raise UrdfException("Sphere has no radius.")
        self.radius = float(sphere["radius"])

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True, convex_hull=True):
        center = tm.get_transform(self.frame, frame)[:3, 3]
        color = self.color if self.color is not None else "k"
        return plot_sphere(
            ax, self.radius, center, wireframe=wireframe, alpha=alpha,
            color=color)


class Cylinder(object):
    def __init__(self, frame, mesh_path, package_dir, color):
        self.frame = frame
        self.color = color
        self.radius = 0.0
        self.length = 0.0

    def parse(self, cylinder):
        if not cylinder.has_attr("radius"):
            raise UrdfException("Cylinder has no radius.")
        self.radius = float(cylinder["radius"])
        if not cylinder.has_attr("length"):
            raise UrdfException("Cylinder has no length.")
        self.length = float(cylinder["length"])

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True, convex_hull=True):
        A2B = tm.get_transform(self.frame, frame)
        color = self.color if self.color is not None else "k"
        return plot_cylinder(
            ax, self.length, self.radius, 0.0, A2B, wireframe=wireframe,
            alpha=alpha, color=color)


class Mesh(object):
    def __init__(self, frame, mesh_path, package_dir, color):
        self.frame = frame
        self.mesh_path = mesh_path
        self.package_dir = package_dir
        self.color = color
        self.filename = None
        self.scale = np.ones(3)

    def parse(self, mesh):
        if self.mesh_path is None and self.package_dir is None:
            self.filename = None
        else:
            if not mesh.has_attr("filename"):
                raise UrdfException("Mesh has no filename.")
            if self.mesh_path is not None:
                self.filename = os.path.join(self.mesh_path, mesh["filename"])
            else:
                assert self.package_dir is not None
                self.filename = mesh["filename"].replace(
                    "package://", self.package_dir)
            if mesh.has_attr("scale"):
                self.scale = np.fromstring(mesh["scale"], sep=" ")

    def plot(self, tm, frame, ax=None, alpha=0.3, wireframe=True, convex_hull=True):
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
