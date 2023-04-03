"""
====
TODO
====
"""
import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.uncertainty as pu
import pytransform3d.visualizer as pv


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
    def __init__(self, robot_urdf, ee_frame, base_frame, joint_names,
                 mesh_path=None, package_dir=None):
        super(ProbabilisticRobotKinematics, self).__init__(check=False)
        self.load_urdf(robot_urdf, mesh_path=mesh_path,
                       package_dir=package_dir)
        self.ee2base_home, self.screw_axes = \
            self._get_screw_axes(ee_frame, base_frame, joint_names)

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

        screw_axes : array, shape (6, n_joints)
            The joint screw axes in the space frame when the manipulator is at
            the home position, in the format of a matrix with axes as the
            columns.
        """
        ee2base_home = self.get_transform(ee_frame, base_frame)
        screw_axes = []
        for jn in joint_names:
            _, ln, _, s_axis, limits, joint_type = self._joints[jn]
            link2base = self.get_transform(ln, base_frame)
            q = link2base[:3, 3]
            if joint_type == "revolute":
                h = 0.0
            elif joint_type == "prismatic":
                h = np.inf
            else:
                raise NotImplementedError(
                    "Joint type %s not supported." % joint_type)
            screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
            screw_axes.append(screw_axis)
        screw_axes = np.column_stack(screw_axes)
        return ee2base_home, screw_axes

    def probabilistic_forward_kinematics(self, thetas, covs):
        """Compute probabilistic forward kinematics.

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
        """
        assert len(thetas) == self.screw_axes.shape[1]
        T = self.ee2base_home
        cov = np.zeros((6, 6))
        Sthetas = self.screw_axes * thetas[np.newaxis]
        Sthetas_transforms = ptr.transforms_from_exponential_coordinates(
            Sthetas.T)
        for i in range(len(thetas) - 1, -1, -1):
            T, cov = pu.concat_locally_uncertain_transforms(
                T, cov, Sthetas_transforms[i], covs[i])
        return T, cov


BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "pytransform3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()

joint_names = ["joint%d" % i for i in range(1, 7)]
tm = ProbabilisticRobotKinematics(
    robot_urdf, "tcp", "linkmount", joint_names, mesh_path=data_dir)

thetas = 0.5 * np.ones(len(joint_names))
for joint_name, theta in zip(joint_names, thetas):
    tm.set_joint(joint_name, theta)

T, cov = tm.probabilistic_forward_kinematics(
    thetas, np.zeros((len(thetas), 6, 6)))

fig = pv.figure()
graph = fig.plot_graph(tm, "robot_arm", show_visuals=True)
fig.plot_transform(T, s=0.1)
fig.view_init()
fig.show()
