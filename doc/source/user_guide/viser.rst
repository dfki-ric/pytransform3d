============================
Viser Visualization Backend
============================

pytransform3d ships with two 3D visualization backends. The default one,
:mod:`pytransform3d.visualizer`, is backed by
`Open3D <http://www.open3d.org/>`_ and opens a native desktop window. The
second backend, :mod:`pytransform3d.viser`, is backed by
`viser <https://viser.studio/>`_ and renders in a web browser. Because the
viser server runs over a WebSocket, the viser backend works in headless
environments (remote servers, containers, notebooks) where a display is not
available.

Both backends share the same public API, so switching between them is a
one-line change.

------------
Installation
------------

viser is an optional dependency. Install it together with the other optional
packages using:

.. code-block:: bash

    python -m pip install 'pytransform3d[all]'

or install viser on its own:

.. code-block:: bash

    python -m pip install viser

`trimesh <https://trimsh.org/>`_ is required for shapes that have no native
viser primitive (cylinders, cones, capsules, ellipsoids, and vectors). It is
included in the ``all`` extras group.

-----------
Basic Usage
-----------

Replace the import of :mod:`pytransform3d.visualizer` with
:mod:`pytransform3d.viser`. Everything else stays the same:

.. code-block:: python

    import numpy as np
    import pytransform3d.viser as pv
    from pytransform3d.transformations import random_transform

    rng = np.random.default_rng(0)

    fig = pv.figure()
    fig.plot_transform(A2B=random_transform(rng))
    fig.plot_sphere(radius=0.3, A2B=np.eye(4), c=(0.2, 0.6, 1.0))
    fig.show()

Calling :func:`~pytransform3d.viser.Figure.show` prints the URL of the viser
server to the console and returns immediately. Open the printed URL in a
browser to view the scene. The server keeps running until the Python process
exits or until you call ``fig._server.stop()``.

Open3D's :func:`~pytransform3d.visualizer.Figure.show` blocks until the
window is closed; the viser :func:`~pytransform3d.viser.Figure.show` does not.
If you need the script to wait for user interaction you can add a blocking
call after ``fig.show()``:

.. code-block:: python

    fig.show()
    input("Press Enter to exit...")

-----------
Differences
-----------

The two backends behave identically except for the following points:

* **Window vs browser.** Open3D opens a native window; viser opens a browser
  tab. After calling :func:`~pytransform3d.viser.Figure.show`, navigate to
  the printed URL (typically ``http://localhost:8080``).

* **show() is non-blocking.** The viser
  :func:`~pytransform3d.viser.Figure.show` returns immediately after printing
  the URL; the Open3D version blocks until the window is closed.

* **save_image() requires playwright.** The viser backend renders the scene
  in a headless Chromium browser to produce the image. Install the extra
  dependencies with ``pip install playwright imageio`` and then run
  ``playwright install chromium`` before calling
  :func:`~pytransform3d.viser.Figure.save_image`.

* **set_line_width() has no effect.** viser does not expose a line-width
  setting after the scene is created. The method issues a
  :class:`UserWarning` and returns without error.

* **port.** The default port is ``8080``. Pass a different port to
  :func:`~pytransform3d.viser.figure` if that port is already in use:

  .. code-block:: python

      fig = pv.figure(port=9090)

---------
Animation
---------

:func:`~pytransform3d.viser.Figure.animate` works the same way as in the
Open3D backend. The callback receives the frame index and any extra arguments
and should return a list of artists. The viser server pushes each frame to
all connected browsers automatically:

.. code-block:: python

    import numpy as np
    import pytransform3d.viser as pv
    from pytransform3d.transformations import transform_from
    from pytransform3d.rotations import matrix_from_axis_angle

    fig = pv.figure()
    frame = fig.plot_transform()
    fig.show()

    n_frames = 60

    def update(step, frame):
        angle = 2.0 * np.pi * step / n_frames
        A2B = transform_from(
            R=matrix_from_axis_angle([0, 0, 1, angle]), p=np.zeros(3)
        )
        frame.set_data(A2B)
        return [frame]

    fig.animate(update, n_frames, loop=True, fargs=(frame,))

--------
Examples
--------

The gallery contains six viser examples that cover a range of use cases:

* :ref:`sphx_glr__auto_examples_viser_vis_viser_shapes.py` -- all geometric
  primitives in a single static scene.
* :ref:`sphx_glr__auto_examples_viser_vis_viser_robot_arm.py` -- animated
  6-DOF robot arm with real-time TCP trajectory tracing.
* :ref:`sphx_glr__auto_examples_viser_vis_viser_wrench_dynamics.py` --
  rigid-body simulation driven by a body-fixed wrench, with a rolling
  position trail.
* :ref:`sphx_glr__auto_examples_viser_vis_viser_camera_orbit.py` -- pinhole
  camera orbiting a static scene, showing the frustum at each pose.
* :ref:`sphx_glr__auto_examples_viser_vis_viser_uncertain_transforms.py` --
  banana distribution from concatenating uncertain transforms: MC-sampled
  paths, mean trajectory, projected SE(3) hyperellipsoid, and position
  ellipsoid.
* :ref:`sphx_glr__auto_examples_viser_vis_viser_probabilistic_robot_kinematics.py`
  -- animated 6-DOF robot with the PPOE end-effector pose uncertainty
  ellipsoid updating in real time.
