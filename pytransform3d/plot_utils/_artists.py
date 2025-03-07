"""Matplotlib artists."""

import numpy as np
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D


class Frame(artist.Artist):
    """A Matplotlib artist that displays a frame represented by its basis.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    draw_label_indicator : bool, optional (default: True)
        Controls whether the line from the frame origin to frame label is
        drawn.

    Other arguments except 'c' and 'color' are passed on to Line3D.
    """

    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(Frame, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.draw_label_indicator = kwargs.pop("draw_label_indicator", True)

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="x")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data(
                np.array([p[0], p[0] + self.s * R[0, d]]),
                np.array([p[1], p[1] + self.s * R[1, d]]),
            )
            b.set_3d_properties(np.array([p[2], p[2] + self.s * R[2, d]]))

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            if self.draw_label_indicator:
                self.label_indicator.set_data(
                    np.array([p[0], label_pos[0]]),
                    np.array([p[1], label_pos[1]]),
                )
                self.label_indicator.set_3d_properties(
                    np.array([p[2], label_pos[2]])
                )

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2], zdir="x")

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer, *args, **kwargs)
        if self.draw_label:
            if self.draw_label_indicator:
                self.label_indicator.draw(renderer, *args, **kwargs)
            self.label_text.draw(renderer, *args, **kwargs)
        super(Frame, self).draw(renderer, *args, **kwargs)

    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            if self.draw_label_indicator:
                axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)


class LabeledFrame(Frame):
    """Displays a frame represented by its basis with axis labels.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    draw_label_indicator : bool, optional (default: True)
        Controls whether the line from the frame origin to frame label is
        drawn.

    Other arguments except 'c' and 'color' are passed on to Line3D.
    """

    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        self.x_label = Text3D(0, 0, 0, text="", zdir="x")
        self.y_label = Text3D(0, 0, 0, text="", zdir="x")
        self.z_label = Text3D(0, 0, 0, text="", zdir="x")
        super(LabeledFrame, self).__init__(A2B, label=label, s=s, **kwargs)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        super(LabeledFrame, self).set_data(A2B, label)

        R = A2B[:3, :3]
        p = A2B[:3, 3]
        x_label_location = p + 1.1 * self.s * R[:, 0]
        y_label_location = p + 1.1 * self.s * R[:, 1]
        z_label_location = p + 1.1 * self.s * R[:, 2]

        self.x_label.set_text("x")
        self.x_label.set_position(x_label_location[:2])
        self.x_label.set_3d_properties(x_label_location[2], zdir="x")

        self.y_label.set_text("y")
        self.y_label.set_position(y_label_location[:2])
        self.y_label.set_3d_properties(y_label_location[2], zdir="x")

        self.z_label.set_text("z")
        self.z_label.set_position(z_label_location[:2])
        self.z_label.set_3d_properties(z_label_location[2], zdir="x")

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        self.x_label.draw(renderer, *args, **kwargs)
        self.y_label.draw(renderer, *args, **kwargs)
        self.z_label.draw(renderer, *args, **kwargs)
        super(LabeledFrame, self).draw(renderer, *args, **kwargs)

    def add_frame(self, axis):
        """Add the frame to a 3D axis."""
        super(LabeledFrame, self).add_frame(axis)
        axis._add_text(self.x_label)
        axis._add_text(self.y_label)
        axis._add_text(self.z_label)


class Trajectory(artist.Artist):
    """A Matplotlib artist that displays a trajectory.

    Parameters
    ----------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices

    show_direction : bool, optional (default: True)
        Plot an arrow to indicate the direction of the trajectory

    n_frames : int, optional (default: 10)
        Number of frames that should be plotted to indicate the rotation

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn

    Other arguments are passed onto Line3D.
    """

    def __init__(self, H, show_direction=True, n_frames=10, s=1.0, **kwargs):
        super(Trajectory, self).__init__()

        self.show_direction = show_direction

        self.trajectory = Line3D([], [], [], **kwargs)
        self.key_frames = [
            Frame(np.eye(4), s=s, **kwargs) for _ in range(n_frames)
        ]

        if self.show_direction:
            self.direction_arrow = Arrow3D(
                [0, 0],
                [0, 0],
                [0, 0],
                mutation_scale=20,
                lw=1,
                arrowstyle="-|>",
                color="k",
            )

        self.set_data(H)

    def set_data(self, H):
        """Set the trajectory data.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Sequence of poses represented by homogeneous matrices
        """
        positions = H[:, :3, 3]
        self.trajectory.set_data(positions[:, 0], positions[:, 1])
        self.trajectory.set_3d_properties(positions[:, 2])

        key_frames_indices = np.linspace(
            0, len(H) - 1, len(self.key_frames), dtype=np.int64
        )
        for i, key_frame_idx in enumerate(key_frames_indices):
            self.key_frames[i].set_data(H[key_frame_idx])

        if self.show_direction:
            start = 0.8 * positions[0] + 0.2 * positions[-1]
            end = 0.2 * positions[0] + 0.8 * positions[-1]
            self.direction_arrow.set_data(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]]
            )

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        self.trajectory.draw(renderer, *args, **kwargs)
        for key_frame in self.key_frames:
            key_frame.draw(renderer, *args, **kwargs)
        if self.show_direction:
            self.direction_arrow.draw(renderer)
        super(Trajectory, self).draw(renderer, *args, **kwargs)

    def add_trajectory(self, axis):
        """Add the trajectory to a 3D axis."""
        axis.add_line(self.trajectory)
        for key_frame in self.key_frames:
            key_frame.add_frame(axis)
        if self.show_direction:
            axis.add_artist(self.direction_arrow)


class Arrow3D(FancyArrowPatch):
    """A Matplotlib patch that represents an arrow in 3D.

    Source: http://stackoverflow.com/a/11156353/915743
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_data(self, xs, ys, zs):
        """Set the arrow data.

        Parameters
        ----------
        xs : iterable
            List of x positions

        ys : iterable
            List of y positions

        zs : iterable
            List of z positions
        """
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the patch."""
        xs3d, ys3d, zs3d = self._verts3d
        try:
            M = self.axes.M
        except AttributeError:
            # deprecated since matplotlib 3.4, will be removed in 3.6
            M = renderer.M
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)

    def do_3d_projection(self, renderer=None):
        # This supports both matplotlib 3.4 and 3.5
        return 0


class Camera(artist.Artist):
    """A Matplotlib artist that displays a camera.

    This function is inspired by Blender's camera visualization. It will
    show the camera center, a virtual image plane, and the top of the virtual
    image plane.

    Parameters
    ----------
    M : array-like, shape (3, 3)
        Intrinsic camera matrix that contains the focal lengths on the diagonal
        and the center of the the image in the last column. It does not matter
        whether values are given in meters or pixels as long as the unit is the
        same as for the sensor size.

    cam2world : array-like, shape (4, 4)
        Transformation matrix of camera in world frame. We assume that the
        position is given in meters.

    virtual_image_distance : float, optional (default: 1)
        Distance from pinhole to virtual image plane that will be displayed.
        We assume that this distance is given in meters.

    sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
        Size of the image sensor: (width, height). It does not matter whether
        values are given in meters or pixels as long as the unit is the same as
        for the sensor size.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha.
    """

    def __init__(
        self,
        M,
        cam2world,
        virtual_image_distance=1.0,
        sensor_size=(1920, 1080),
        **kwargs,
    ):
        super(Camera, self).__init__()

        if "c" in kwargs:
            color = kwargs.pop("c")
        elif "color" in kwargs:
            color = kwargs.pop("color")
        else:
            color = "k"

        self.sensor_corners = _calculate_sensor_corners_in_camera(
            M, virtual_image_distance, sensor_size
        )
        self.top_corners = _calculate_top_corners_in_camera(self.sensor_corners)

        self.lines_sensor = [
            Line3D([], [], [], color=color, **kwargs) for _ in range(4)
        ]
        self.line_top = Line3D([], [], [], color=color, **kwargs)

        self.set_data(cam2world)

    def set_data(self, cam2world):
        """Set the transformation data.

        Parameters
        ----------
        cam2world : array-like, shape (4, 4)
            Transform from frame A to frame B
        """
        cam2world = np.asarray(cam2world)
        sensor_in_world = np.dot(
            cam2world,
            np.vstack(
                (self.sensor_corners.T, np.ones(len(self.sensor_corners)))
            ),
        )
        for i in range(4):
            xs, ys, zs = [
                [
                    cam2world[j, 3],
                    sensor_in_world[j, i],
                    sensor_in_world[j, (i + 1) % 4],
                ]
                for j in range(3)
            ]
            self.lines_sensor[i].set_data_3d(xs, ys, zs)

        top_in_world = np.dot(
            cam2world,
            np.vstack((self.top_corners.T, np.ones(len(self.top_corners)))),
        )
        xs, ys, zs, _ = top_in_world
        self.line_top.set_data_3d(xs, ys, zs)

    @artist.allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        """Draw the artist."""
        for b in self.lines_sensor:
            b.draw(renderer, *args, **kwargs)
        self.line_top.draw(renderer, *args, **kwargs)
        super(Camera, self).draw(renderer, *args, **kwargs)

    def add_camera(self, axis):
        """Add the camera to a 3D axis."""
        for b in self.lines_sensor:
            axis.add_line(b)
        axis.add_line(self.line_top)


def _calculate_sensor_corners_in_camera(M, virtual_image_distance, sensor_size):
    """Calculate the corners of the sensor frame in camera coordinates."""
    focal_length = np.mean((M[0, 0], M[1, 1]))
    sensor_corners = np.array(
        [
            [0, 0, focal_length],
            [0, sensor_size[1], focal_length],
            [sensor_size[0], sensor_size[1], focal_length],
            [sensor_size[0], 0, focal_length],
        ]
    )
    sensor_corners[:, 0] -= M[0, 2]
    sensor_corners[:, 1] -= M[1, 2]
    return virtual_image_distance / focal_length * sensor_corners


def _calculate_top_corners_in_camera(sensor_corners):
    """Calculate the corners of the top triangle in camera coordinates."""
    up = sensor_corners[0] - sensor_corners[1]
    return np.array(
        [
            sensor_corners[0] + 0.1 * up,
            0.5 * (sensor_corners[0] + sensor_corners[3]) + 0.5 * up,
            sensor_corners[3] + 0.1 * up,
            sensor_corners[0] + 0.1 * up,
        ]
    )
