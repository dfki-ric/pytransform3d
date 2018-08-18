==========================
Transformation Ambiguities
==========================

There are lots of ambiguities in the world of transformations. We try to
explain them all here.

-------------------------------------------------
Active (Alibi) vs. Passive (Alias) Transformation
-------------------------------------------------

An active transformation

* changes the physical position of an object
* can be defined in the absence of a coordinate system or does not change the
  current coordinate system
* is used exclusively by mathematicians

Another name for active transformation is alibi transformation.

A passive transformation

* changes the coordinate system in which the object is described
* does not change the object
* could be used by physicists and engineers (e.g. roboticists)

Another name for passive transformation is alias transformation.

The following illustration compares the active view with the passive view.
The position of the data is interpreted in the frame indicated by solid
axes.
We use exactly the same transformation matrix in both plots.
In the active view, we see that the transformation is applied to the data.
The data is physically moved. The dashed basis represents a frame that is
moved from the base frame with the same transformation. The data is
now interpreted in the old frame.
The the passive we move the frame with the transformation. The data stays
at its original position but it is interpreted in the new frame.

.. raw:: html

    <table>
    <tr><td>Active</td><td>Passive</td></tr>
    <tr>
    <td>

.. plot::
    :width: 400px

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from pytransform.transformations import transform, plot_transform
    from pytransform.plot_utils import Arrow3D


    plt.figure()
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-1.05, 1.05), ylim=(-0.55, 1.55), zlim=(-1.05, 1.05),
                xlabel="X", ylabel="Y", zlabel="Z")
    ax.view_init(elev=90, azim=-90)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())

    random_state = np.random.RandomState(42)
    PA = np.ones((10, 4))
    PA[:, :3] = 0.1 * random_state.randn(10, 3)
    PA[:, 0] += 0.3
    PA[:, :3] += 0.3

    x_translation = -0.1
    y_translation = 0.2
    z_rotation = np.pi / 4.0
    A2B = np.array([
        [np.cos(z_rotation), -np.sin(z_rotation), 0.0, x_translation],
        [np.sin(z_rotation), np.cos(z_rotation), 0.0, y_translation],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    PB = transform(A2B, PA)

    plot_transform(ax=ax, A2B=np.eye(4))
    ax.scatter(PA[:, 0], PA[:, 1], PA[:, 2], c="orange")
    plot_transform(ax=ax, A2B=A2B, ls="--", alpha=0.5)
    ax.scatter(PB[:, 0], PB[:, 1], PB[:, 2], c="cyan")

    axis_arrow = Arrow3D(
        [0.7, 0.3],
        [0.4, 0.9],
        [0.2, 0.2],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(axis_arrow)

    plt.tight_layout()
    plt.show()

.. raw:: html

    </td>
    <td>

.. plot::
    :width: 400px

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d
    from pytransform.transformations import transform, plot_transform
    from pytransform.plot_utils import Arrow3D


    plt.figure()
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-1.05, 1.05), ylim=(-0.55, 1.55), zlim=(-1.05, 1.05),
                xlabel="X", ylabel="Y", zlabel="Z")
    ax.view_init(elev=90, azim=-90)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())

    random_state = np.random.RandomState(42)
    PA = np.ones((10, 4))
    PA[:, :3] = 0.1 * random_state.randn(10, 3)
    PA[:, 0] += 0.3
    PA[:, :3] += 0.3

    x_translation = -0.1
    y_translation = 0.2
    z_rotation = np.pi / 4.0
    A2B = np.array([
        [np.cos(z_rotation), -np.sin(z_rotation), 0.0, x_translation],
        [np.sin(z_rotation), np.cos(z_rotation), 0.0, y_translation],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    plot_transform(ax=ax, A2B=np.eye(4), ls="--", alpha=0.5)
    ax.scatter(PA[:, 0], PA[:, 1], PA[:, 2], c="orange")
    plot_transform(ax=ax, A2B=A2B)

    axis_arrow = Arrow3D(
        [0.0, -0.1],
        [0.0, 0.2],
        [0.2, 0.2],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(axis_arrow)

    plt.tight_layout()
    plt.show()

.. raw:: html

    </td>
    </tr>
    <table>

Using the inverse transformation in the active view gives us exactly the same
solution as the original transformation in the passive view and vice versa.

It is usually easy to determine whether the active or the passive convention
is used by taking a look at the rotation matrix: when we rotate
counter-clockwise by an angle :math:`\theta` about the z-axis, the following
rotation matrix is usually used in an active transformation:

.. math::

    \left( \begin{array}{ccc}
        \cos \theta & -\sin \theta & 0 \\
        \sin \theta & \cos \theta & 0 \\
        0 & 0 & 1\\
    \end{array} \right)

Its transformed version is usually used for a passive transformation:

.. math::

    \left( \begin{array}{ccc}
        \cos \theta & \sin \theta & 0 \\
        -\sin \theta & \cos \theta & 0 \\
        0 & 0 & 1\\
    \end{array} \right)

.. note::

    The default in pytransform are passive transformations.

Reference:

Selig, J.M.: Active Versus Passive Transformations in Robotics, 2006,
IEEE Robotics and Automation Magazine.
PDF: https://core.ac.uk/download/pdf/77055186.pdf.

-------------------------------------
Source in Target vs. Source to Target
-------------------------------------

A transformations is defined between two frames: source and target, origin
and target, or parent and child. You can also say it transforms from some
frame to another frame.
However, it is not always clear how to interpret the source and the target.

**Source in target:**
The first option would be to assume that the source is the base frame in
which we represent the target frame and the transformation gives us the
translation and rotation to get the location of the target in the source.
In the illustration below that would mean that the object (target) is
defined in the camera frame (source) and the camera (target) is defined in
the body frame (source).

**Source to target:**
The second option assumes that the transformation transforms data, for example,
points from the source frame to the target frame. In the illustration below
that would mean we have a transformation to transform points from the
object frame (source) to the camera frame (target) and a transformation
to transform points from the camera frame (source) to the body frame (target).

Whenever you hear that there is a transformation from some frame to another
frame, make sure you understand what is meant.

.. note::

    The default in pytransform is source in target.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    from pytransform.rotations import random_quaternion, q_id
    from pytransform.transformations import transform_from_pq
    from pytransform.transform_manager import TransformManager


    random_state = np.random.RandomState(0)

    camera2body = transform_from_pq(
        np.hstack((np.array([0.4, -0.3, 0.5]),
                   random_quaternion(random_state))))
    object2camera = transform_from_pq(
        np.hstack((np.array([0.0, 0.0, 0.3]),
                   random_quaternion(random_state))))

    tm = TransformManager()
    tm.add_transform("camera", "body", camera2body)
    tm.add_transform("object", "camera", object2camera)

    ax = tm.plot_frames_in("body", s=0.1)
    ax.set_xlim((-0.15, 0.65))
    ax.set_ylim((-0.4, 0.4))
    ax.set_zlim((0.0, 0.8))
    plt.show()


--------------------------------
Right-multiply vs. Left-multiply
--------------------------------



-----------------------
Intrinsic vs. Extrinsic
-----------------------



----------------------------------------------
Right-handed vs. Left-handed Coordinate System
----------------------------------------------

We typically use a right-handed coordinate system, that is, the x-, y- and
z-axis are aligned in a specific way. The name comes from the way how the
fingers are attached to the human hand. Try to align your thumb with the
imaginary x-axis, your index finger with the y-axis, and your middle finger
with the z-axis. It is possible to do this with a right hand in a
right-handed system and with the left hand in a left-handed system.

.. raw:: html

    <table>
    <tr><td>Right-handed</td><td>Left-handed</td></tr>
    <tr>
    <td>

.. plot::
    :width: 400px

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d


    plt.figure()
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), zlim=(-0.05, 1.05),
            xlabel="X", ylabel="Y", zlabel="Z")

    basis = np.eye(3)
    for d, c in enumerate(["r", "g", "b"]):
        ax.plot([0.0, basis[0, d]],
                [0.0, basis[1, d]],
                [0.0, basis[2, d]], color=c, lw=5)

    plt.show()

.. raw:: html

    </td>
    <td>

.. plot::
    :width: 400px

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d


    plt.figure()
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), zlim=(-1.05, 0.05),
            xlabel="X", ylabel="Y", zlabel="Z")

    basis = np.eye(3)
    basis[:, 2] *= -1.0
    for d, c in enumerate(["r", "g", "b"]):
        ax.plot([0.0, basis[0, d]],
                [0.0, basis[1, d]],
                [0.0, basis[2, d]], color=c, lw=5)

    plt.show()

.. raw:: html

    </td>
    </tr>
    <table>