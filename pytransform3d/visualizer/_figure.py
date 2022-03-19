"""Figure based on Open3D's visualizer."""
import numpy as np
import open3d as o3d
from .. import rotations as pr
from .. import transformations as pt
from .. import trajectories as ptr
from ._artists import (Line3D, PointCollection3D, Vector3D, Frame, Trajectory,
                       Camera, Box, Sphere, Cylinder, Mesh, Ellipsoid, Capsule,
                       Cone, Plane, Graph)


class Figure:
    """The top level container for all the plot elements.

    You can close the visualizer with the keys `escape` or `q`.

    Parameters
    ----------
    window_name : str, optional (default: Open3D)
        Window title name.

    width : int, optional (default: 1920)
        Width of the window.

    height : int, optional (default: 1080)
        Height of the window.

    with_key_callbacks : bool, optional (default: False)
        Creates a visualizer that allows to register callbacks
        for keys.
    """
    def __init__(self, window_name="Open3D", width=1920, height=1080,
                 with_key_callbacks=False):
        if with_key_callbacks:
            self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        else:
            self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(
            window_name=window_name, width=width, height=height)

    def add_geometry(self, geometry):
        """Add geometry to visualizer.

        Parameters
        ----------
        geometry : Geometry
            Open3D geometry.
        """
        self.visualizer.add_geometry(geometry)

    def _remove_geometry(self, geometry):
        """Remove geometry to visualizer.

        .. warning::

            This function is not public because the interface of the
            underlying visualizer might change in the future causing the
            signature of this function to change as well.

        Parameters
        ----------
        geometry : Geometry
            Open3D geometry.
        """
        self.visualizer.remove_geometry(geometry)

    def update_geometry(self, geometry):
        """Indicate that geometry has been updated.

        Parameters
        ----------
        geometry : Geometry
            Open3D geometry.
        """
        self.visualizer.update_geometry(geometry)

    def remove_artist(self, artist):
        """Remove artist from visualizer.

        Parameters
        ----------
        artist : Artist
            Artist that should be removed from this figure.
        """
        for g in artist.geometries:
            self._remove_geometry(g)

    def set_line_width(self, line_width):
        """Set render option line width.

        Note: this feature does not work in Open3D's visualizer at the
        moment.

        Parameters
        ----------
        line_width : float
            Line width.
        """
        self.visualizer.get_render_option().line_width = line_width
        self.visualizer.update_renderer()

    def set_zoom(self, zoom):
        """Set zoom.

        Parameters
        ----------
        zoom : float
            Zoom of the visualizer.
        """
        self.visualizer.get_view_control().set_zoom(zoom)

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
            When callback does not return any artists
        """
        initialized = False
        window_open = True
        while window_open and (loop or not initialized):
            for i in range(n_frames):
                drawn_artists = callback(i, *fargs)

                if drawn_artists is None:
                    raise RuntimeError(
                        "The animation function must return a "
                        "sequence of Artist objects.")
                try:
                    drawn_artists = [a for a in drawn_artists]
                except TypeError:
                    drawn_artists = [drawn_artists]

                for a in drawn_artists:
                    for geometry in a.geometries:
                        self.update_geometry(geometry)

                window_open = self.visualizer.poll_events()
                if not window_open:
                    break
                self.visualizer.update_renderer()
            initialized = True

    def view_init(self, azim=-60, elev=30):
        """Set the elevation and azimuth of the axes.

        Parameters
        ----------
        azim : float, optional (default: -60)
            Azimuth angle in the x,y plane in degrees.

        elev : float, optional (default: 30)
            Elevation angle in the z plane.
        """
        vc = self.visualizer.get_view_control()
        pcp = vc.convert_to_pinhole_camera_parameters()
        distance = np.linalg.norm(pcp.extrinsic[:3, 3])
        R_azim_elev_0_world2camera = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]])
        R_azim_elev_0_camera2world = R_azim_elev_0_world2camera.T
        # azimuth and elevation are defined in world frame
        R_azim = pr.active_matrix_from_angle(2, np.deg2rad(azim))
        R_elev = pr.active_matrix_from_angle(1, np.deg2rad(-elev))
        R_elev_azim_camera2world = R_azim.dot(R_elev).dot(
            R_azim_elev_0_camera2world)
        pcp.extrinsic = pt.transform_from(  # world2camera
            R=R_elev_azim_camera2world.T,
            p=[0, 0, distance])
        vc.convert_from_pinhole_camera_parameters(pcp)

    def plot(self, P, c=(0, 0, 0)):
        """Plot line.

        Parameters
        ----------
        P : array-like, shape (n_points, 3)
            Points of which the line consists.

        c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
            Color can be given as individual colors per line segment or
            as one color for each segment. A color is represented by 3
            values between 0 and 1 indicate representing red, green, and
            blue respectively.

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
            Scaling of the spheres that will be drawn.

        c : array-like, shape (3,) or (n_points, 3), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicate
            representing red, green, and blue respectively.

        Returns
        -------
        point_collection : PointCollection3D
            New point collection.
        """
        point_collection = PointCollection3D(P, s, c)
        point_collection.add_artist(self)
        return point_collection

    def plot_vector(self, start=np.zeros(3), direction=np.array([1, 0, 0]),
                    c=(0, 0, 0)):
        """Plot vector.

        Parameters
        ----------
        start : array-like, shape (3,), optional (default: [0, 0, 0])
            Start of the vector

        direction : array-like, shape (3,), optional (default: [1, 0, 0])
            Direction of the vector

        c : array-like, shape (3,), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicate
            representing red, green, and blue respectively.

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
        Frame : frame
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
        Frame : frame
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
            A color is represented by 3 values between 0 and 1 indicate
            representing red, green, and blue respectively.

        Returns
        -------
        trajectory : Trajectory
            New trajectory.
        """
        H = ptr.matrices_from_pos_quat(P)
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

    def plot_cylinder(self, length=2.0, radius=1.0, A2B=np.eye(4),
                      resolution=20, split=4, c=None):
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
            The height will be split into split segments

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

    def plot_ellipsoid(self, radii=np.ones(3), A2B=np.eye(4), resolution=20,
                       c=None):
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

    def plot_capsule(self, height=1, radius=1, A2B=np.eye(4), resolution=20,
                     c=None):
        """Plot capsule.

        A capsule is the volume covered by a sphere moving along a line segment.

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

    def plot_cone(self, height=1, radius=1, A2B=np.eye(4), resolution=20,
                  c=None):
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

    def plot_plane(self, normal=np.array([0.0, 0.0, 1.0]), d=None,
                   point_in_plane=None, s=1.0, c=None):
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
            self, tm, frame, show_frames=False, show_connections=False,
            show_visuals=False, show_collision_objects=False,
            show_name=False, whitelist=None, s=1.0):
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

        s : float, optional (default: 1)
            Scaling of the frames that will be drawn

        Returns
        -------
        graph : Graph
            New graph.
        """
        graph = Graph(tm, frame, show_frames, show_connections,
                      show_visuals, show_collision_objects, show_name,
                      whitelist, s)
        graph.add_artist(self)
        return graph

    def plot_camera(self, M, cam2world=None, virtual_image_distance=1,
                    sensor_size=(1920, 1080), strict_check=True):
        """Plot camera in world coordinates.

        This function is inspired by Blender's camera visualization. It will
        show the camera center, a virtual image plane, and the top of the
        virtual image plane.

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
        camera = Camera(M, cam2world, virtual_image_distance, sensor_size,
                        strict_check)
        camera.add_artist(self)
        return camera

    def save_image(self, filename):
        """Save rendered image to file.

        Parameters
        ----------
        filename : str
            Path to file in which the rendered image should be stored
        """
        self.visualizer.capture_screen_image(filename, True)

    def show(self):
        """Display the figure window."""
        self.visualizer.run()
        self.visualizer.destroy_window()


def figure(window_name="Open3D", width=1920, height=1080,
           with_key_callbacks=False):
    """Create a new figure.

    Parameters
    ----------
    window_name : str, optional (default: Open3D)
        Window title name.

    width : int, optional (default: 1920)
        Width of the window.

    height : int, optional (default: 1080)
        Height of the window.

    with_key_callbacks : bool, optional (default: False)
        Creates a visualizer that allows to register callbacks
        for keys.

    Returns
    -------
    figure : Figure
        New figure.
    """
    return Figure(window_name, width, height, with_key_callbacks)
