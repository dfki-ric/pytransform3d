import numpy as np
import open3d as o3d
from . import rotations as pr
from . import transformations as pt
from . import trajectories as ptr
from . import urdf
from itertools import chain
import warnings


# TODO docstrings


def figure(window_name="Open3D"):
    return Figure(window_name)


class Frame:
    def __init__(self, A2B, label=None, s=1.0):
        self.A2B = None
        self.label = None
        self.s = s

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.s)

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B
        self.label = label
        if label is not None:
            warnings.warn(
                "This viewer does not support text. Frame label "
                "will be ignored.")

        self.frame.transform(pt.concat(pt.invert_transform(previous_A2B), self.A2B))

    def add_frame(self, figure):
        figure.add_geometry(self.frame)

    @property
    def geometries(self):
        return [self.frame]


def plot_basis(figure, R=None, p=np.zeros(3), s=1.0, strict_check=True):
    if R is None:
        R = np.eye(3)
    R = pr.check_matrix(R, strict_check=strict_check)

    frame = Frame(pt.transform_from(R=R, p=p), s=s)
    frame.add_frame(figure)

    return frame


def plot_transform(figure, A2B=None, s=1.0, ax_s=1, name=None, strict_check=True):
    if A2B is None:
        A2B = np.eye(4)
    A2B = pt.check_transform(A2B, strict_check=strict_check)

    frame = Frame(A2B, name, s)
    frame.add_frame(figure)

    return frame


class Trajectory:
    def __init__(self, H, show_direction=True, n_frames=10, s=1.0, c=[0, 0, 0]):
        self.H = H
        self.show_direction = show_direction
        self.n_frames = n_frames
        self.s = s
        self.c = c

        self.key_frames = []
        self.line_set = o3d.geometry.LineSet()

        self.key_frames_indices = np.linspace(
            0, len(self.H) - 1, self.n_frames, dtype=np.int)
        for i, key_frame_idx in enumerate(self.key_frames_indices):
            self.key_frames.append(Frame(self.H[key_frame_idx], s=self.s))

        self.set_data(H)

    def set_data(self, H):
        assert not self.show_direction
        self.line_set.points = o3d.utility.Vector3dVector(H[:, :3, 3])
        self.line_set.lines = o3d.utility.Vector2iVector(np.hstack((
            np.arange(len(H) - 1)[:, np.newaxis],
            np.arange(1, len(H))[:, np.newaxis])))
        self.line_set.colors = o3d.utility.Vector3dVector(
            [self.c for _ in range(len(H))])
        for i, key_frame_idx in enumerate(self.key_frames_indices):
            self.key_frames[i].set_data(H[key_frame_idx])

    def add_trajectory(self, figure):
        figure.add_geometry(self.line_set)
        for key_frame in self.key_frames:
            key_frame.add_frame(figure)

    @property
    def geometries(self):
        frame_geometries = list(
            chain(*[kf.geometries for kf in self.key_frames]))
        return [self.line_set] + frame_geometries


def plot_trajectory(figure, P, show_direction=True, n_frames=10, s=1.0, c=[0, 0, 0]):
    H = ptr.matrices_from_pos_quat(P)
    assert not show_direction, "not implemented yet"
    trajectory = Trajectory(H, show_direction, n_frames, s, c)
    trajectory.add_trajectory(figure)
    return trajectory


class Mesh:
    def __init__(self, filename, A2B=np.eye(4), s=np.ones(3), c=None):
        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.mesh.vertices) * s)
        if c is not None:
            n_vertices = len(self.mesh.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.mesh.transform(pt.concat(pt.invert_transform(previous_A2B), self.A2B))

    def add_artist(self, figure):
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        return [self.mesh]


def plot_mesh(figure, filename, A2B=np.eye(4), s=np.ones(3), c=None):
    mesh = Mesh(filename, A2B, s, c)
    mesh.add_artist(figure)
    return mesh


class Cylinder:
    def __init__(self, length=2.0, radius=1.0, A2B=np.eye(4), resolution=20, split=4, c=None):
        self.cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=resolution, split=split)
        if c is not None:
            n_vertices = len(self.cylinder.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.cylinder.transform(pt.concat(pt.invert_transform(previous_A2B), self.A2B))

    def add_artist(self, figure):
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        return [self.cylinder]


def plot_cylinder(figure, length=2.0, radius=1.0, A2B=np.eye(4), resolution=20, split=4, c=None):
    cylinder = Cylinder(length, radius, A2B, resolution, split, c)
    cylinder.add_artist(figure)
    return cylinder


class Box:
    def __init__(self, size=np.ones(3), A2B=np.eye(4), c=None):
        self.half_size = np.asarray(size) / 2.0
        width, height, depth = size
        self.box = o3d.geometry.TriangleMesh.create_box(
            width, height, depth)
        if c is not None:
            n_vertices = len(self.box.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.box.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.box.transform(
            pt.concat(pt.transform_from(R=np.eye(3), p=-self.half_size),
                      pt.concat(pt.invert_transform(previous_A2B), self.A2B)))

    def add_artist(self, figure):
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        return [self.box]


def plot_box(figure, size=np.ones(3), A2B=np.eye(4), c=None):
    box = Box(size, A2B, c)
    box.add_artist(figure)
    return box


class Sphere:
    def __init__(self, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius, resolution)
        if c is not None:
            n_vertices = len(self.sphere.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.sphere.transform(pt.concat(pt.invert_transform(previous_A2B), self.A2B))

    def add_artist(self, figure):
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        return [self.sphere]


def plot_sphere(figure, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
    sphere = Sphere(radius, A2B, resolution, c)
    sphere.add_artist(figure)
    return sphere


class Graph:
    def __init__(self, tm, frame, show_frames=False, show_connections=False, show_name=False, whitelist=None, s=1.0, c=(0, 0, 0)):
        self.tm = tm
        self.frame = frame
        self.show_frames = show_frames
        self.show_connections = show_connections
        self.whitelist = whitelist
        self.s = s
        self.c = c

        if self.frame not in self.tm.nodes:
            raise KeyError("Unknown frame '%s'" % self.frame)

        self.nodes = list(sorted(self.tm._whitelisted_nodes(whitelist)))

        self.frames = {}
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, frame)
                    name = node if show_name else None
                    self.frames[node] = Frame(node2frame, name, s)
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        self.connections = {}
        if self.show_connections:
            for frame_names in self.tm.transforms.keys():
                from_frame, to_frame = frame_names
                if from_frame in self.tm.nodes and to_frame in self.tm.nodes:
                    try:
                        self.tm.get_transform(from_frame, self.frame)
                        self.tm.get_transform(to_frame, self.frame)
                        self.connections[frame_names] = o3d.geometry.LineSet()
                    except KeyError:
                        pass  # Frame is not connected to the reference frame
        self.set_data()

    def set_data(self):
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, self.frame)
                    self.frames[node].set_data(node2frame)
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        if self.show_connections:
            for frame_names in self.connections.keys():
                from_frame, to_frame = frame_names
                try:
                    from2ref = self.tm.get_transform(from_frame, self.frame)
                    to2ref = self.tm.get_transform(to_frame, self.frame)

                    points = np.vstack((from2ref[:3, 3], to2ref[:3, 3]))
                    self.connections[frame_names].points = o3d.utility.Vector3dVector(points)
                    self.connections[frame_names].lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
                    self.connections[frame_names].colors = o3d.utility.Vector3dVector([self.c for _ in range(2)])
                except KeyError:
                    pass  # Frame is not connected to the reference frame

    def add_artist(self, figure):  # TODO move to base class
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        geometries = []
        if self.show_frames:
            for f in self.frames.values():
                geometries += f.geometries
        if self.show_connections:
            geometries += list(self.connections.values())
        return geometries


def plot_graph(figure, tm, frame, show_frames=False, show_connections=False, show_name=False, whitelist=None, s=1.0, c=(0, 0, 0)):
    graph = Graph(tm, frame, show_frames, show_connections, show_name, whitelist, s, c)
    graph.add_artist(figure)
    return graph


class Figure:
    def __init__(self, window_name="Open3D"):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window(window_name=window_name)

    def add_geometry(self, geometry):
        self.visualizer.add_geometry(geometry)

    def update_geometry(self, geometry):
        self.visualizer.update_geometry(geometry)

    def set_line_width(self, line_width):
        self.visualizer.get_render_option().line_width = line_width
        self.visualizer.update_renderer()

    def set_zoom(self, zoom):
        self.visualizer.get_view_control().set_zoom(zoom)

    def animate(self, callback, n_frames, loop=False, fargs=()):
        initialized = False
        window_open = True
        while window_open and (loop or not initialized):
            for i in range(n_frames):
                drawn_artists = callback(i, *fargs)

                if drawn_artists is None:
                    raise RuntimeError('The animation function must return a '
                                       'sequence of Artist objects.')
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
        R_elev_azim_camera2world = R_azim.dot(R_elev).dot(R_azim_elev_0_camera2world)
        pcp.extrinsic = pt.transform_from(  # world2camera
            R=R_elev_azim_camera2world.T,
            p=[0, 0, distance])
        vc.convert_from_pinhole_camera_parameters(pcp)

    def show(self):
        self.visualizer.run()
        self.visualizer.destroy_window()


Figure.plot_basis = plot_basis
Figure.plot_trajectory = plot_trajectory
Figure.plot_mesh = plot_mesh
Figure.plot_transform = plot_transform
Figure.plot_cylinder = plot_cylinder
Figure.plot_box = plot_box
Figure.plot_sphere = plot_sphere
Figure.plot_graph = plot_graph


def show_urdf_transform_manager(
        figure, tm, frame, collision_objects=False, visuals=False,
        frames=False, whitelist=None, s=1.0, c=None):
    if collision_objects:
        if hasattr(tm, "collision_objects"):
            _add_objects(figure, tm, tm.collision_objects, frame, c)
    if visuals:
        if hasattr(tm, "visuals"):
            _add_objects(figure, tm, tm.visuals, frame, c)
    if frames:
        for node in tm.nodes:
            _add_frame(figure, tm, node, frame, whitelist, s)


def _add_objects(figure, tm, objects, frame, c=None):
    for obj in objects:
        obj.show(figure, tm, frame, c)


def _add_frame(figure, tm, from_frame, to_frame, whitelist=None, s=1.0):
    if whitelist is not None and from_frame not in whitelist:
        return
    A2B = tm.get_transform(from_frame, to_frame)
    frame = Frame(A2B, s=s)
    frame.add_frame(figure)


def box_show(self, figure, tm, frame, c=None):
    raise NotImplementedError()


urdf.Box.show = box_show


def sphere_show(self, figure, tm, frame, c=None):
    raise NotImplementedError()


urdf.Sphere.show = sphere_show


def cylinder_show(self, figure, tm, frame, c=None):
    raise NotImplementedError()


urdf.Cylinder.show = cylinder_show


def mesh_show(self, figure, tm, frame, c=None):  # TODO refactor
    if self.mesh_path is None:
        print("No mesh path given")
        return
    A2B = tm.get_transform(self.frame, frame)

    scale = self.scale
    mesh = o3d.io.read_triangle_mesh(self.filename)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * scale)
    mesh.transform(A2B)
    if c is not None:
        n_vertices = len(mesh.vertices)
        colors = np.zeros((n_vertices, 3))
        colors[:] = c
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    figure.add_geometry(mesh)


urdf.Mesh.show = mesh_show
