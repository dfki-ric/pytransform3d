"""
================================
Concatenate Uncertain Transforms
================================

In this example, we assume that a robot is moving with constant velocity
along the x-axis, however, there is noise in the orientation of the robot
that accumulates and leads to different paths when sampling. Uncertainty
accumulation leads to the so-called banana distribution, which does not seem
Gaussian in Cartesian space, but it is Gaussian in exponential coordinates
of SE(3).

This example is adapted and modified to 3D from Barfoot and Furgale [1]_.
The banana distribution was analyzed in detail by Long et al. [2]_.
"""

import numpy as np
import trimesh

import pytransform3d.rotations as pr
import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
import pytransform3d.viser as pv

# %%
# Configuration
# -------------
# We assume :math:`\Delta t = 1 s` and constant velocity. The covariance
# models rotational noise that accumulates over the trajectory.
rng = np.random.default_rng(0)
cov_pose_chol = np.diag([0, 0.02, 0.03, 0, 0, 0])
cov_pose = np.dot(cov_pose_chol, cov_pose_chol.T)
velocity_vector = np.array([0, 0, 0, 1.0, 0, 0])
T_vel = pt.transform_from_exponential_coordinates(velocity_vector)
n_steps = 100
n_mc_samples = 1000

# %%
# Estimated mean trajectory and accumulated covariance
# -----------------------------------------------------
T_est = np.eye(4)
path = np.zeros((n_steps + 1, 6))
path[0] = pt.exponential_coordinates_from_transform(T_est)
cov_est = np.zeros((6, 6))
for t in range(n_steps):
    T_est, cov_est = pu.concat_globally_uncertain_transforms(
        T_est, cov_est, T_vel, cov_pose
    )
    path[t + 1] = pt.exponential_coordinates_from_transform(T_est)

# %%
# Monte-Carlo sampling
# --------------------
T = np.eye(4)
mc_path = np.zeros((n_steps + 1, n_mc_samples, 4, 4))
mc_path[0, :] = T
for t in range(n_steps):
    noise_samples = ptr.transforms_from_exponential_coordinates(
        cov_pose_chol.dot(rng.standard_normal(size=(6, n_mc_samples))).T
    )
    step_samples = ptr.concat_many_to_one(noise_samples, T_vel)
    mc_path[t + 1] = np.einsum("nij,njk->nik", step_samples, mc_path[t])
mc_path_vec = np.einsum(
    "tinm,tin->tim", mc_path[:, :, :3, :3], mc_path[:, :, :3, 3]
)

# %%
# Surface helper
# --------------


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


# %%
# Scene setup
# -----------
fig = pv.figure()
# The scene extends ~100 units along the x-axis and ~30 units along y.
# azim=0, elev=0 places the camera at +z from center, looking along -z so
# that x goes right and y goes up — the banana in the x-y plane is fully
# visible.  distance=120 is large enough to frame the whole trajectory.
fig.view_init(azim=0, elev=0, center=(50.0, 0.0, 5.0), distance=120.0)

# %%
# MC-sampled trajectories (every 10th path to keep rendering fast)
# -----------------------------------------------------------------
# A light blue colour approximates the low-opacity effect used in the
# matplotlib version.
for i in range(0, n_mc_samples, 10):
    fig.plot(mc_path_vec[:, i, :], c=(0.6, 0.75, 1.0))

# %%
# Scatter of final MC positions
# ------------------------------
fig.scatter(mc_path_vec[-1], s=0.4, c=(0.0, 0.0, 0.8))

# %%
# Mean trajectory with coordinate frames
# ----------------------------------------
fig.plot_trajectory(
    ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(path)),
    n_frames=10,
    s=5.0,
)

# %%
# Projected hyperellipsoid of the final pose distribution
# --------------------------------------------------------
# This is the "banana" shape: the 3D projection of the equiprobable
# hyper-ellipsoid of the Gaussian in SE(3) tangent space.
x, y, z = pu.to_projected_ellipsoid(T_est, cov_est, factor=3.0, n_steps=50)
banana_verts, banana_faces = grid_to_mesh(x, y, z)
yellow = (230, 210, 20)
fig.scene.add_mesh_simple(
    name="/banana/solid",
    vertices=banana_verts,
    faces=banana_faces,
    color=yellow,
    opacity=0.3,
    side="double",
)
fig.scene.add_mesh_simple(
    name="/banana/wireframe",
    vertices=banana_verts,
    faces=banana_faces,
    color=yellow,
    wireframe=True,
    side="double",
)

# %%
# Position-only ellipsoid from MC samples
# ----------------------------------------
# This is the ellipsoid fitted to the spread of the final positions of the
# sampled trajectories in Cartesian space.
mean_mc = np.mean(mc_path_vec[-1], axis=0)
cov_mc = np.cov(mc_path_vec[-1], rowvar=False)
ellipsoid2origin, radii = pu.to_ellipsoid(mean_mc, cov_mc)
ell_mesh = trimesh.creation.icosphere(subdivisions=3)
ell_mesh.vertices = ell_mesh.vertices * (3.0 * radii[np.newaxis])
ell_verts = ell_mesh.vertices.astype(np.float32)
ell_faces = np.array(ell_mesh.faces, dtype=np.int32)
ell_wxyz = pr.quaternion_from_matrix(
    ellipsoid2origin[:3, :3], strict_check=False
)
ell_pos = ellipsoid2origin[:3, 3].astype(np.float32)
magenta = (210, 30, 210)
fig.scene.add_mesh_simple(
    name="/pos_ellipsoid/solid",
    vertices=ell_verts,
    faces=ell_faces,
    color=magenta,
    opacity=0.1,
    side="double",
    wxyz=ell_wxyz,
    position=ell_pos,
)
fig.scene.add_mesh_simple(
    name="/pos_ellipsoid/wireframe",
    vertices=ell_verts,
    faces=ell_faces,
    color=magenta,
    wireframe=True,
    side="double",
    wxyz=ell_wxyz,
    position=ell_pos,
)

if "__file__" in globals():
    fig.show()
    input("Press Enter to exit...")

# %%
# References
# ----------
# .. [1] Barfoot, T. D., Furgale, P. T. (2014). Associating Uncertainty With
#    Three-Dimensional Poses for Use in Estimation Problems. IEEE Transactions
#    on Robotics 30(3), pp. 679-693, doi: 10.1109/TRO.2014.2298059.
#
# .. [2] Long, A. W., Wolfe, K. C., Mashner, M. J., Chirikjian, G. S. (2013).
#    The Banana Distribution is Gaussian: A Localization Study with Exponential
#    Coordinates. In Robotics: Science and Systems VIII, pp. 265-272.
#    http://www.roboticsproceedings.org/rss08/p34.pdf
