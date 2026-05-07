"""
=====================================
Probabilistic Product of Exponentials
=====================================

We compute the probabilistic forward kinematics of a robot with flexible
links or joints and visualize the projected equiprobably ellipsoid of the
end-effector's pose distribution.
"""

import os

import numpy as np

import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
import pytransform3d.viser as pv
from pytransform3d.urdf import UrdfTransformManager


# %%
# Probabilistic Robot Kinematics
# ------------------------------
#
# The end-effector's pose distribution is computed based on the Probabilistic
# Product of Exponentials PPOE [1]_.
#
# Our ProbabilisticRobotKinematics class is a subclass of
# :class:`~pytransform3d.urdf.UrdfTransformManager`, which loads a description
# of a robot from the URDF format.
#
# The complicated part of this example is the conversion of kinematics
# parameters from URDF data to screw axes that are needed for the product
# of exponentials formulation of forward kinematics.
#
# Once we have this information, the implementation of the probabilistic
# product of exponentials is straightforward:
#
# 1. We multiply the screw axis of each joint with the corresponding joint
#    angle to obtain the exponential coordinates of each relative joint
#    displacement.
# 2. We concatenate the relative joint displacements and the base pose to
#    obtain the end-effector's pose. This is the original product of
#    exponentials.
# 3. The PPOE modifies the original product of exponentials by transforming
#    and concatenating the covariances of each transformation.
class ProbabilisticRobotKinematics(UrdfTransformManager):
    """Probabilistic robot kinematics.

    Parameters
    ----------
    robot_urdf : str
        URDF description of robot

    ee_frame : str
        Name of the end-effector frame

    base_frame : str
        Name of the base frame

    joint_names : list
        Names of joints in order from base to end effector

    mesh_path : str, optional (default: None)
        Path in which we search for meshes that are defined in the URDF.
        Meshes will be ignored if it is set to None and no 'package_dir'
        is given.

    package_dir : str, optional (default: None)
        Some URDFs start file names with 'package://' to refer to the ROS
        package in which these files (textures, meshes) are located. This
        variable defines to which path this prefix will be resolved.
    """

    def __init__(
        self,
        robot_urdf,
        ee_frame,
        base_frame,
        joint_names,
        mesh_path=None,
        package_dir=None,
    ):
        super(ProbabilisticRobotKinematics, self).__init__(check=False)
        self.load_urdf(robot_urdf, mesh_path=mesh_path, package_dir=package_dir)
        self.ee2base_home, self.screw_axes_home = self._get_screw_axes(
            ee_frame, base_frame, joint_names
        )
        self.joint_limits = np.array(
            [self.get_joint_limits(jn) for jn in joint_names]
        )

    def _get_screw_axes(self, ee_frame, base_frame, joint_names):
        """Get screw axes of joints in space frame at robot's home position.

        Parameters
        ----------
        ee_frame : str
            Name of the end-effector frame

        base_frame : str
            Name of the base frame

        joint_names : list
            Names of joints in order from base to end effector

        Returns
        -------
        ee2base_home : array, shape (4, 4)
            The home configuration (position and orientation) of the
            end-effector.

        screw_axes_home : array, shape (n_joints, 6)
            The joint screw axes in the space frame when the manipulator is at
            the home position.
        """
        ee2base_home = self.get_transform(ee_frame, base_frame)
        screw_axes_home = []
        for jn in joint_names:
            ln, _, _, s_axis, limits, joint_type = self._joints[jn]
            link2base = self.get_transform(ln, base_frame)
            s_axis = np.dot(link2base[:3, :3], s_axis)
            q = link2base[:3, 3]

            if joint_type == "revolute":
                h = 0.0
            elif joint_type == "prismatic":
                h = np.inf
            else:
                raise NotImplementedError(
                    "Joint type %s not supported." % joint_type
                )

            screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
            screw_axes_home.append(screw_axis)
        screw_axes_home = np.row_stack(screw_axes_home)
        return ee2base_home, screw_axes_home

    def probabilistic_forward_kinematics(self, thetas, covs):
        """Compute probabilistic forward kinematics.

        This is based on the probabilistic product of exponentials.

        Parameters
        ----------
        thetas : array, shape (n_joints,)
            A list of joint coordinates.

        covs : array, shape (n_joints, 6, 6)
            Covariances of joint transformations.

        Returns
        -------
        ee2base : array, shape (4, 4)
            A homogeneous transformation matrix representing the end-effector
            frame when the joints are at the specified coordinates.

        cov : array, shape (6, 6)
            Covariance of the pose in tangent space.
        """
        assert len(thetas) == self.screw_axes_home.shape[0]
        thetas = np.clip(
            thetas, self.joint_limits[:, 0], self.joint_limits[:, 1]
        )

        Sthetas = self.screw_axes_home * thetas[:, np.newaxis]
        joint_displacements = ptr.transforms_from_exponential_coordinates(
            Sthetas
        )

        T = np.eye(4)
        cov = np.zeros((6, 6))
        for i in range(len(thetas)):
            T, cov = pu.concat_locally_uncertain_transforms(
                joint_displacements[i], T, covs[i], cov
            )

        T = T.dot(self.ee2base_home)
        ad = pt.adjoint_from_transform(self.ee2base_home)
        cov = ad.dot(cov).dot(ad.T)

        return T, cov


# %%
# Surface helper
# --------------
# To visualize the 6D covariance in the tangent space of SE(3), we project its
# equiprobable hyper-ellipsoid to 3D and represent it as a mesh.


def grid_to_mesh(x, y, z):
    """Triangulate a surface defined by grid arrays.

    Parameters
    ----------
    x : array, shape (m, n)
        x-coordinates of the surface grid.

    y : array, shape (m, n)
        y-coordinates of the surface grid.

    z : array, shape (m, n)
        z-coordinates of the surface grid.

    Returns
    -------
    vertices : array, shape (m*n, 3)
        Vertex positions.

    faces : array, shape ((m-1)*(n-1)*2, 3)
        Triangle face indices.
    """
    m, n = x.shape
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()]).astype(
        np.float32
    )
    triangles = []
    for i in range(m - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v11 = (i + 1) * n + j + 1
            v01 = i * n + j + 1
            triangles.extend([[v00, v10, v11], [v00, v11, v01]])
    return vertices, np.array(triangles, dtype=np.int32)


class Surface:
    """Surface mesh to be visualized with viser.

    Parameters
    ----------
    scene : viser scene handle
        The viser scene to add the mesh to.

    name : str
        Base name for the mesh objects in the viser scene.

    x : array, shape (n_steps, n_steps)
        Coordinates on x-axis of grid on surface.

    y : array, shape (n_steps, n_steps)
        Coordinates on y-axis of grid on surface.

    z : array, shape (n_steps, n_steps)
        Coordinates on z-axis of grid on surface.

    c : array-like, shape (3,), optional (default: None)
        RGB color in [0, 1] range.
    """

    def __init__(self, scene, name, x, y, z, c=None):
        self._scene = scene
        self._name = name
        if c is not None:
            arr = np.asarray(c, dtype=float)
            self._color = tuple(int(np.clip(v * 255, 0, 255)) for v in arr[:3])
        else:
            self._color = (0, 128, 128)
        self.set_data(x, y, z)

    def set_data(self, x, y, z):
        """Update the surface mesh.

        Parameters
        ----------
        x : array, shape (n_steps, n_steps)
            Coordinates on x-axis of grid on surface.

        y : array, shape (n_steps, n_steps)
            Coordinates on y-axis of grid on surface.

        z : array, shape (n_steps, n_steps)
            Coordinates on z-axis of grid on surface.
        """
        vertices, faces = grid_to_mesh(x, y, z)
        self._scene.add_mesh_simple(
            name=self._name + "/solid",
            vertices=vertices,
            faces=faces,
            color=self._color,
            opacity=0.5,
            side="double",
        )
        self._scene.add_mesh_simple(
            name=self._name + "/wire",
            vertices=vertices,
            faces=faces,
            color=self._color,
            wireframe=True,
            side="double",
        )


# %%
# Animation callback
# ------------------
# Each frame updates the joint angles and recomputes the end-effector pose
# distribution, then refreshes both the robot graph and the ellipsoid mesh.
def animation_callback(
    step, n_frames, tm, graph, joint_names, thetas, covs, surface
):
    angle = 0.5 * np.cos(2.0 * np.pi * (0.5 + step / n_frames))
    thetas_t = angle * thetas
    for joint_name, value in zip(joint_names, thetas_t):
        tm.set_joint(joint_name, value)
    graph.set_data()

    T, cov = tm.probabilistic_forward_kinematics(thetas_t, covs)
    x, y, z = pu.to_projected_ellipsoid(T, cov, factor=1, n_steps=50)
    surface.set_data(x, y, z)

    return graph, surface


# %%
# Setup
# -----
# We load the URDF file,
BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (
    not os.path.exists(data_dir)
    and os.path.dirname(search_path) != "pytransform3d"
):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()

# %%
# define the kinematic chain that we are interested in,
joint_names = ["joint%d" % i for i in range(1, 7)]
tm = ProbabilisticRobotKinematics(
    robot_urdf, "tcp", "linkmount", joint_names, mesh_path=data_dir
)

# %%
# define the joint angles,
thetas = np.array([1, 1, 1, 0, 1, 0])
current_thetas = -0.5 * thetas
for joint_name, theta in zip(joint_names, current_thetas):
    tm.set_joint(joint_name, theta)

# %%
# and define the covariances of the joints.
covs = np.zeros((len(thetas), 6, 6))
covs[0] = np.diag([0, 0, 1, 0, 0, 0])
covs[1] = np.diag([0, 1, 0, 0, 0, 0])
covs[2] = np.diag([0, 1, 0, 0, 0, 0])
covs[4] = np.diag([0, 1, 0, 0, 0, 0])
covs *= 0.05

# %%
# PPOE and Visualization
# ----------------------
#
# Then we can finally use PPOE to compute the end-effector pose and its
# covariance.
T, cov = tm.probabilistic_forward_kinematics(current_thetas, covs)

# %%
# We compute the 3D projection of the 6D covariance matrix.
x, y, z = pu.to_projected_ellipsoid(T, cov, factor=1, n_steps=50)

# %%
# The following code sets up and animates the visualization.
fig = pv.figure()
graph = fig.plot_graph(tm, "robot_arm", show_visuals=True)
fig.plot_transform(np.eye(4), s=0.3)
surface = Surface(fig.scene, "/ee_ellipsoid", x, y, z, c=(0, 0.5, 0.5))
fig.view_init(elev=20, azim=0)
n_frames = 200
if "__file__" in globals():
    fig.show()
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(n_frames, tm, graph, joint_names, thetas, covs, surface),
    )
    input("Press Enter to exit...")

# %%
# References
# ----------
#
# .. [1] Meyer, Strobl, Triebel (2022): The Probabilistic Robot Kinematics
#    Model and its Application to Sensor Fusion. In IEEE/RSJ International
#    Conference on Intelligent Robots and Systems (IROS), Kyoto, Japan, 2022,
#    pp. 3263-3270, doi: 10.1109/IROS47612.2022.9981399.
#    https://elib.dlr.de/191928/1/202212_ELIB_PAPER_VERSION_with_copyright.pdf
