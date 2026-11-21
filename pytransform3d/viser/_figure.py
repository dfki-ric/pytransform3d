"""Figure based on viser."""

import time
import warnings

import numpy as np
import viser

from ._artists import (
    Line3D,
    PointCollection3D,
    Vector3D,
    Frame,
    Trajectory,
    Camera,
    Box,
    Sphere,
    Cylinder,
    Mesh,
    Ellipsoid,
    Capsule,
    Cone,
    Plane,
    Graph,
)
from .. import rotations as pr
from .. import trajectories as ptr
from .. import transformations as pt


class Figure:
    """The top level container for all the plot elements.

    The figure starts a local web server. Open the printed URL in a browser
    to view the scene. Call :func:`show` to print the URL.

    Parameters
    ----------
    window_name : str, optional (default: pytransform3d)
        Label shown at the top of the GUI panel in the browser.

    port : int, optional (default: 8080)
        Port used by the viser web server.
    """

    def __init__(self, window_name="pytransform3d", port=8080):
        self._server = viser.ViserServer(label=window_name, port=port)
        self.scene = self._server.scene
        self.scene.configure_default_lights()
        self.scene.add_grid(name="/_grid", width=10, height=10)
        self._object_count = 0

    def _next_name(self, prefix):
        name = "/%s/%05d" % (prefix, self._object_count)
        self._object_count += 1
        return name

    def remove_artist(self, artist):
        """Remove artist from the scene.

        Parameters
        ----------
        artist : Artist
            Artist that should be removed from this figure.
        """
        artist.remove()

    def set_line_width(self, line_width):
        """Set render option line width.

        Note: this setting is not effective after line segments have been
        added to the scene.

        Parameters
        ----------
        line_width : float
            Line width.
        """
        warnings.warn(
            "set_line_width() has no effect in the viser backend. Pass "
            "line_width when creating the artist.",
            UserWarning,
            stacklevel=2,
        )

    def set_zoom(self, zoom):
        """Set zoom for all connected clients.

        Parameters
        ----------
        zoom : float
            Zoom factor. Values greater than 1 zoom in, less than 1 zoom out.
        """
        for _, client in self._server.get_clients().items():
            pos = np.asarray(client.camera.position)
            client.camera.position = pos / zoom

    def animate(self, callback, n_frames, loop=False, fargs=()):
        """Make animation with callback.

        Parameters
        ----------
        callback : callable
            Callback that will be called in a loop to update geometries.
            The first input of the function will be the current frame
            index from [0, `n_frames`). Further arguments can be given as
            `fargs`. The function should return one artist object or a
            list of artists that have been updated.

        n_frames : int
            Total number of frames.

        loop : bool, optional (default: False)
            Run callback in an infinite loop.

        fargs : list, optional (default: [])
            Arguments that will be passed to the callback.

        Raises
        ------
        RuntimeError
            When callback does not return any artists.
        """
        initialized = False
        while loop or not initialized:
            for i in range(n_frames):
                drawn_artists = callback(i, *fargs)

                if drawn_artists is None:
                    raise RuntimeError(
                        "The animation function must return a "
                        "sequence of Artist objects."
                    )

                time.sleep(1.0 / 30.0)
            initialized = True

    def view_init(
        self, azim=-60, elev=30, center=(0.0, 0.0, 0.0), distance=5.0
    ):
        """Set the initial camera pose for all current and future clients.

        The callback registered here fires for every new browser connection, so
        the view is consistent regardless of when the browser is opened.

        Parameters
        ----------
        azim : float, optional (default: -60)
            Azimuth angle around the world-up axis (y) in degrees. 0 places
            the camera on the +z side of *center*; 90 places it on the +x
            side.

        elev : float, optional (default: 30)
            Elevation angle above the ground plane (x-z) in degrees. 0 is
            level; 90 is directly above *center*.

        center : array-like, shape (3,), optional (default: [0, 0, 0])
            The point the camera looks at.

        distance : float, optional (default: 5)
            Distance from *center* to the camera.
        """
        center = np.asarray(center, dtype=float)
        # viser uses a y-up coordinate system. Azimuth rotates around the
        # y-axis (world up); elevation tilts from the x-z ground plane.
        # Only position and look_at are set; viser derives the camera
        # orientation from those two using y as world-up.
        R_azim = pr.active_matrix_from_angle(1, np.deg2rad(azim))
        R_elev = pr.active_matrix_from_angle(0, np.deg2rad(-elev))
        R = R_azim.dot(R_elev)
        position = center + R.dot(np.array([0.0, 0.0, distance]))

        @self._server.on_client_connect
        def _set_camera(client):
            client.camera.position = position
            client.camera.look_at = center

    def plot(self, P, c=(0, 0, 0)):
        """Plot line.

        Parameters
        ----------
        P : array-like, shape (n_points, 3)
            Points of which the line consists.

        c : array-like, shape (n_points - 1, 3) or (3,), optional
            (default: black). Color can be given as individual colors per
            line segment or as one color for each segment. A color is
            represented by 3 values between 0 and 1 indicating red, green,
            and blue respectively.

        Returns
        -------
        line : Line3D
            New line.
        """
        line3d = Line3D(P, c)
        line3d.add_artist(self)
        return line3d

    def scatter(self, P, s=0.05, c=None):
        """Plot collection of points.

        Parameters
        ----------
        P : array, shape (n_points, 3)
            Points

        s : float, optional (default: 0.05)
            Scaling of the points that will be drawn.

        c : array-like, shape (3,) or (n_points, 3), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicating
            red, green, and blue respectively.

        Returns
        -------
        point_collection : PointCollection3D
            New point collection.
        """
        point_collection = PointCollection3D(P, s, c)
        point_collection.add_artist(self)
        return point_collection

    def plot_vector(
        self, start=np.zeros(3), direction=np.array([1, 0, 0]), c=(0, 0, 0)
    ):
        """Plot vector.

        Parameters
        ----------
        start : array-like, shape (3,), optional (default: [0, 0, 0])
            Start of the vector

        direction : array-like, shape (3,), optional (default: [1, 0, 0])
            Direction of the vector

        c : array-like, shape (3,), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicating
            red, green, and blue respectively.

        Returns
        -------
        vector : Vector3D
            New vector.
        """
        vector3d = Vector3D(start, direction, c)
        vector3d.add_artist(self)
        return vector3d

    def plot_basis(self, R=None, p=np.zeros(3), s=1.0, strict_check=True):
        """Plot basis.

        Parameters
        ----------
        R : array-like, shape (3, 3), optional (default: I)
            Rotation matrix, each column contains a basis vector

        p : array-like, shape (3,), optional (default: [0, 0, 0])
            Offset from the origin

        s : float, optional (default: 1)
            Scaling of the frame that will be drawn

        strict_check : bool, optional (default: True)
            Raise a ValueError if the rotation matrix is not numerically
            close enough to a real rotation matrix. Otherwise we print a
            warning.

        Returns
        -------
        frame : Frame
            New frame.
        """
        if R is None:
            R = np.eye(3)
        R = pr.check_matrix(R, strict_check=strict_check)

        frame = Frame(pt.transform_from(R=R, p=p), s=s)
        frame.add_artist(self)
        return frame

    def plot_transform(self, A2B=None, s=1.0, name=None, strict_check=True):
        """Plot coordinate frame.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        s : float, optional (default: 1)
            Length of basis vectors

        name : str, optional (default: None)
            Name of the frame

        strict_check : bool, optional (default: True)
            Raise a ValueError if the transformation matrix is not
            numerically close enough to a real transformation matrix.
            Otherwise we print a warning.

        Returns
        -------
        frame : Frame
            New frame.
        """
        if A2B is None:
            A2B = np.eye(4)
        A2B = pt.check_transform(A2B, strict_check=strict_check)

        frame = Frame(A2B, name, s)
        frame.add_artist(self)
        return frame

    def plot_trajectory(self, P, n_frames=10, s=1.0, c=(0, 0, 0)):
        """Trajectory of poses.

        Parameters
        ----------
        P : array-like, shape (n_steps, 7), optional (default: None)
            Sequence of poses represented by positions and quaternions in
            the order (x, y, z, w, vx, vy, vz) for each step

        n_frames : int, optional (default: 10)
            Number of frames that should be plotted to indicate the
            rotation

        s : float, optional (default: 1)
            Scaling of the frames that will be drawn

        c : array-like, shape (3,), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicating
            red, green, and blue respectively.

        Returns
        -------
        trajectory : Trajectory
            New trajectory.
        """
        H = ptr.transforms_from_pqs(P)
        trajectory = Trajectory(H, n_frames, s, c)
        trajectory.add_artist(self)
        return trajectory

    def plot_sphere(self, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
        """Plot sphere.

        Parameters
        ----------
        radius : float, optional (default: 1)
            Radius of the sphere

        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        resolution : int, optional (default: 20)
            The resolution of the sphere. The longitudes will be split into
            resolution segments (i.e. there are resolution + 1 latitude
            lines including the north and south pole). The latitudes will
            be split into 2 * resolution segments (i.e. there are
            2 * resolution longitude lines.)

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        sphere : Sphere
            New sphere.
        """
        sphere = Sphere(radius, A2B, resolution, c)
        sphere.add_artist(self)
        return sphere

    def plot_box(self, size=np.ones(3), A2B=np.eye(4), c=None):
        """Plot box.

        Parameters
        ----------
        size : array-like, shape (3,), optional (default: [1, 1, 1])
            Size of the box per dimension

        A2B : array-like, shape (4, 4), optional (default: I)
            Center of the box

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        box : Box
            New box.
        """
        box = Box(size, A2B, c)
        box.add_artist(self)
        return box

    def plot_cylinder(
        self,
        length=2.0,
        radius=1.0,
        A2B=np.eye(4),
        resolution=20,
        split=4,
        c=None,
    ):
        """Plot cylinder.

        Parameters
        ----------
        length : float, optional (default: 1)
            Length of the cylinder.

        radius : float, optional (default: 1)
            Radius of the cylinder.

        A2B : array-like, shape (4, 4)
            Pose of the cylinder. The position corresponds to the center of the
            line segment and the z-axis to the direction of the line segment.

        resolution : int, optional (default: 20)
            The circle will be split into resolution segments

        split : int, optional (default: 4)
            This parameter is ignored. It is accepted for API compatibility
            with the Open3D backend.

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        cylinder : Cylinder
            New cylinder.
        """
        cylinder = Cylinder(length, radius, A2B, resolution, split, c)
        cylinder.add_artist(self)
        return cylinder

    def plot_mesh(self, filename, A2B=np.eye(4), s=np.ones(3), c=None):
        """Plot mesh.

        Parameters
        ----------
        filename : str
            Path to mesh file

        A2B : array-like, shape (4, 4)
            Center of the mesh

        s : array-like, shape (3,), optional (default: [1, 1, 1])
            Scaling of the mesh that will be drawn

        c : array-like, shape (n_vertices, 3) or (3,), optional (default: None)
            Color(s)

        Returns
        -------
        mesh : Mesh
            New mesh.
        """
        mesh = Mesh(filename, A2B, s, c)
        mesh.add_artist(self)
        return mesh

    def plot_ellipsoid(
        self, radii=np.ones(3), A2B=np.eye(4), resolution=20, c=None
    ):
        """Plot ellipsoid.

        Parameters
        ----------
        radii : array-like, shape (3,)
            Radii along the x-axis, y-axis, and z-axis of the ellipsoid.

        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        resolution : int, optional (default: 20)
            The resolution of the ellipsoid. The longitudes will be split into
            resolution segments (i.e. there are resolution + 1 latitude
            lines including the north and south pole). The latitudes will
            be split into 2 * resolution segments (i.e. there are
            2 * resolution longitude lines.)

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        ellipsoid : Ellipsoid
            New ellipsoid.
        """
        ellipsoid = Ellipsoid(radii, A2B, resolution, c)
        ellipsoid.add_artist(self)
        return ellipsoid

    def plot_capsule(
        self, height=1, radius=1, A2B=np.eye(4), resolution=20, c=None
    ):
        """Plot capsule.

        A capsule is the volume covered by a sphere moving along a line
        segment.

        Parameters
        ----------
        height : float, optional (default: 1)
            Height of the capsule along its z-axis.

        radius : float, optional (default: 1)
            Radius of the capsule.

        A2B : array-like, shape (4, 4)
            Pose of the capsule. The position corresponds to the center of the
            line segment and the z-axis to the direction of the line segment.

        resolution : int, optional (default: 20)
            The resolution of the capsule. The longitudes will be split into
            resolution segments (i.e. there are resolution + 1 latitude lines
            including the north and south pole). The latitudes will be split
            into 2 * resolution segments (i.e. there are 2 * resolution
            longitude lines.)

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        capsule : Capsule
            New capsule.
        """
        capsule = Capsule(height, radius, A2B, resolution, c)
        capsule.add_artist(self)
        return capsule

    def plot_cone(
        self, height=1, radius=1, A2B=np.eye(4), resolution=20, c=None
    ):
        """Plot cone.

        Parameters
        ----------
        height : float, optional (default: 1)
            Height of the cone along its z-axis.

        radius : float, optional (default: 1)
            Radius of the cone.

        A2B : array-like, shape (4, 4)
            Pose of the cone.

        resolution : int, optional (default: 20)
            The circle will be split into resolution segments.

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        cone : Cone
            New cone.
        """
        cone = Cone(height, radius, A2B, resolution, c)
        cone.add_artist(self)
        return cone

    def plot_plane(
        self,
        normal=np.array([0.0, 0.0, 1.0]),
        d=None,
        point_in_plane=None,
        s=1.0,
        c=None,
    ):
        """Plot plane.

        Parameters
        ----------
        normal : array-like, shape (3,), optional (default: [0, 0, 1])
            Plane normal.

        d : float, optional (default: None)
            Distance to origin in Hesse normal form.

        point_in_plane : array-like, shape (3,), optional (default: None)
            Point in plane.

        s : float, optional (default: 1)
            Scaling of the plane that will be drawn.

        c : array-like, shape (3,), optional (default: None)
            Color.

        Returns
        -------
        plane : Plane
            New plane.
        """
        plane = Plane(normal, d, point_in_plane, s, c)
        plane.add_artist(self)
        return plane

    def plot_graph(
        self,
        tm,
        frame,
        show_frames=False,
        show_connections=False,
        show_visuals=False,
        show_collision_objects=False,
        show_name=False,
        whitelist=None,
        convex_hull_of_collision_objects=False,
        s=1.0,
    ):
        """Plot graph of connected frames.

        Parameters
        ----------
        tm : TransformManager
            Representation of the graph

        frame : str
            Name of the base frame in which the graph will be displayed

        show_frames : bool, optional (default: False)
            Show coordinate frames

        show_connections : bool, optional (default: False)
            Draw lines between frames of the graph

        show_visuals : bool, optional (default: False)
            Show visuals that are stored in the graph

        show_collision_objects : bool, optional (default: False)
            Show collision objects that are stored in the graph

        show_name : bool, optional (default: False)
            Show names of frames

        whitelist : list, optional (default: all)
            List of frames that should be displayed

        convex_hull_of_collision_objects : bool, optional (default: False)
            Show convex hull of collision objects.

        s : float, optional (default: 1)
            Scaling of the frames that will be drawn

        Returns
        -------
        graph : Graph
            New graph.
        """
        graph = Graph(
            tm,
            frame,
            show_frames,
            show_connections,
            show_visuals,
            show_collision_objects,
            show_name,
            whitelist,
            convex_hull_of_collision_objects,
            s,
        )
        graph.add_artist(self)
        return graph

    def plot_camera(
        self,
        M,
        cam2world=None,
        virtual_image_distance=1,
        sensor_size=(1920, 1080),
        strict_check=True,
    ):
        """Plot camera in world coordinates.

        Parameters
        ----------
        M : array-like, shape (3, 3)
            Intrinsic camera matrix that contains the focal lengths on the
            diagonal and the center of the the image in the last column. It
            does not matter whether values are given in meters or pixels as
            long as the unit is the same as for the sensor size.

        cam2world : array-like, shape (4, 4), optional (default: I)
            Transformation matrix of camera in world frame. We assume that the
            position is given in meters.

        virtual_image_distance : float, optional (default: 1)
            Distance from pinhole to virtual image plane that will be
            displayed. We assume that this distance is given in meters. The
            unit has to be consistent with the unit of the position in
            cam2world.

        sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
            Size of the image sensor: (width, height). It does not matter
            whether values are given in meters or pixels as long as the unit is
            the same as for the sensor size.

        strict_check : bool, optional (default: True)
            Raise a ValueError if the transformation matrix is not numerically
            close enough to a real transformation matrix. Otherwise we print a
            warning.

        Returns
        -------
        camera : Camera
            New camera.
        """
        camera = Camera(
            M, cam2world, virtual_image_distance, sensor_size, strict_check
        )
        camera.add_artist(self)
        return camera

    def save_image(self, filename):
        """Save rendered image to file.

        Launches a headless Chromium browser via Playwright, connects it to
        the viser server, waits for the scene to render, and saves the result.

        Requires ``playwright`` and ``imageio``::

            pip install playwright imageio
            playwright install chromium

        Parameters
        ----------
        filename : str
            Path to file in which the rendered image should be stored.
            The extension determines the format (e.g. ``.jpg``, ``.png``).

        Raises
        ------
        ImportError
            If ``playwright`` or ``imageio`` are not installed.
        RuntimeError
            If no browser client connects within the timeout.
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise ImportError(
                "save_image() requires playwright. "
                "Install with: pip install playwright "
                "&& playwright install chromium"
            ) from exc
        try:
            import imageio
        except ImportError as exc:
            raise ImportError(
                "save_image() requires imageio: pip install imageio"
            ) from exc

        port = self._server.get_port()
        url = f"http://localhost:{port}"

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--use-gl=swiftshader",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(url)

            # Poll until the browser's WebSocket client registers.
            timeout = 10.0
            start = time.time()
            while not self._server.get_clients():
                if time.time() - start > timeout:
                    browser.close()
                    raise RuntimeError(
                        "Viser browser client did not connect within "
                        f"{timeout:.0f} s."
                    )
                time.sleep(0.1)

            # Allow extra time for Three.js to process all scene messages.
            page.wait_for_timeout(3000)

            client = next(iter(self._server.get_clients().values()))
            image = client.get_render(
                height=720, width=1280, transport_format="jpeg"
            )
            imageio.imwrite(filename, image)
            browser.close()

    def show(self):
        """Print the URL to open in a browser.

        The viser server runs in a background thread and stays alive until
        the Python process ends or :attr:`_server` is stopped manually.
        """
        port = self._server.get_port()
        print("Open in browser: http://localhost:%d" % port)


def figure(window_name="pytransform3d", port=8080):
    """Create a new figure.

    Parameters
    ----------
    window_name : str, optional (default: pytransform3d)
        Label shown at the top of the GUI panel in the browser.

    port : int, optional (default: 8080)
        Port used by the viser web server.

    Returns
    -------
    figure : Figure
        New figure.
    """
    return Figure(window_name, port)
