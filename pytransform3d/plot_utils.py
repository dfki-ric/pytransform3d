"""Utilities for plotting."""
import numpy as np
import warnings
try:
    import matplotlib.pyplot as plt
    from matplotlib import artist
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    from mpl_toolkits.mplot3d.art3d import Line3D, Text3D, Poly3DCollection, Line3DCollection
    from .transformations import transform
    from .rotations import unitx, unitz, perpendicular_to_vectors, norm_vector


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

        Other arguments except 'c' and 'color' are passed on to Line3D.
        """
        def __init__(self, A2B, label=None, s=1.0, **kwargs):
            super(Frame, self).__init__()

            if "c" in kwargs:
                kwargs.pop("c")
            if "color" in kwargs:
                kwargs.pop("color")

            self.s = s

            self.x_axis = Line3D([], [], [], color="r", **kwargs)
            self.y_axis = Line3D([], [], [], color="g", **kwargs)
            self.z_axis = Line3D([], [], [], color="b", **kwargs)

            self.draw_label = label is not None
            self.label = label

            if self.draw_label:
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
                b.set_data(np.array([p[0], p[0] + self.s * R[0, d]]),
                           np.array([p[1], p[1] + self.s * R[1, d]]))
                b.set_3d_properties(np.array([p[2], p[2] + self.s * R[2, d]]))

            if self.draw_label:
                if label is None:
                    label = self.label
                label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

                self.label_indicator.set_data(
                    np.array([p[0], label_pos[0]]),
                    np.array([p[1], label_pos[1]]))
                self.label_indicator.set_3d_properties(
                    np.array([p[2], label_pos[2]]))

                self.label_text.set_text(label)
                self.label_text.set_position([label_pos[0], label_pos[1]])
                self.label_text.set_3d_properties(label_pos[2], zdir="x")

        @artist.allow_rasterization
        def draw(self, renderer, *args, **kwargs):
            """Draw the artist."""
            for b in [self.x_axis, self.y_axis, self.z_axis]:
                b.draw(renderer, *args, **kwargs)
            if self.draw_label:
                self.label_indicator.draw(renderer, *args, **kwargs)
                self.label_text.draw(renderer, *args, **kwargs)
            super(Frame, self).draw(renderer, *args, **kwargs)

        def add_frame(self, axis):
            """Add the frame to a 3D axis."""
            for b in [self.x_axis, self.y_axis, self.z_axis]:
                axis.add_line(b)
            if self.draw_label:
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

        Other arguments except 'c' and 'color' are passed on to Line3D.
        """
        def __init__(self, A2B, label=None, s=1.0, **kwargs):
            self.x_label = Text3D(0, 0, 0, text="", zdir="x")
            self.y_label = Text3D(0, 0, 0, text="", zdir="x")
            self.z_label = Text3D(0, 0, 0, text="", zdir="x")
            super(LabeledFrame, self).__init__(A2B, label=None, s=1.0, **kwargs)

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
            self.key_frames = [Frame(np.eye(4), s=s, **kwargs)
                               for _ in range(n_frames)]

            if self.show_direction:
                self.direction_arrow = Arrow3D(
                    [0, 0], [0, 0], [0, 0],
                    mutation_scale=20, lw=1, arrowstyle="-|>", color="k")

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
                0, len(H) - 1, len(self.key_frames), dtype=np.int)
            for i, key_frame_idx in enumerate(key_frames_indices):
                self.key_frames[i].set_data(H[key_frame_idx])

            if self.show_direction:
                start = 0.8 * positions[0] + 0.2 * positions[-1]
                end = 0.2 * positions[0] + 0.8 * positions[-1]
                self.direction_arrow.set_data(
                    [start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

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


    class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
        """A Matplotlib patch that represents an arrow in 3D."""
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
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            super(Arrow3D, self).draw(renderer)


    def make_3d_axis(ax_s, pos=111):
        """Generate new 3D axis.

        Parameters
        ----------
        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        pos : int, optional (default: 111)
            Position indicator (nrows, ncols, plot_number)

        Returns
        -------
        ax : Matplotlib 3d axis
            New axis
        """
        try:
            ax = plt.subplot(pos, projection="3d", aspect="equal")
        except NotImplementedError:
            # HACK: workaround for bug in new matplotlib versions (ca. 3.02):
            # "It is not currently possible to manually set the aspect"
            ax = plt.subplot(pos, projection="3d")
        plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
                 xlabel="X", ylabel="Y", zlabel="Z")
        return ax


    def plot_vector(ax=None, start=np.zeros(3), direction=np.array([1, 0, 0]), s=1.0, arrowstyle="simple", ax_s=1, **kwargs):
        """Plot Vector.

        Draws an arrow from start to start + s * direction.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        start : array-like, shape (3,), optional (default: [0, 0, 0])
            Start of the vector

        direction : array-like, shape (3,), optional (default: [0, 0, 0])
            Direction of the vector

        s : float, optional (default: 1)
            Scaling of the vector that will be drawn

        arrowstyle : str, or ArrowStyle, optional (default: 'simple')
            See matplotlib's documentation of arrowstyle in
            matplotlib.patches.FancyArrowPatch for more options

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        kwargs : dict, optional (default: {})
            Additional arguments for the plotting functions, e.g. alpha

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        axis_arrow = Arrow3D(
            [start[0], start[0] + s * direction[0]],
            [start[1], start[1] + s * direction[1]],
            [start[2], start[2] + s * direction[2]],
            mutation_scale=20, arrowstyle=arrowstyle, **kwargs)
        ax.add_artist(axis_arrow)

        return ax


    def plot_length_variable(ax=None, start=np.zeros(3), end=np.ones(3), name="l", above=False, ax_s=1, color="k", **kwargs):
        """Plot length with text at its center.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        start : array-like, shape (3,), optional (default: [0, 0, 0])
            Start point

        end : array-like, shape (3,), optional (default: [1, 1, 1])
            End point

        name : str, optional (default: 'l')
            Text in the middle

        above : bool, optional (default: False)
            Plot name above line

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        color : str, optional (default: black)
            Color in which the cylinder should be plotted

        kwargs : dict, optional (default: {})
            Additional arguments for the text, e.g. fontsize
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        direction = end - start
        length = np.linalg.norm(direction)

        if above:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
        else:
            mid1 = start + 0.4 * direction
            mid2 = start + 0.6 * direction
            ax.plot([start[0], mid1[0]], [start[1], mid1[1]], [start[2], mid1[2]], color=color)
            ax.plot([end[0], mid2[0]], [end[1], mid2[1]], [end[2], mid2[2]], color=color)

        if np.linalg.norm(direction / length - unitz) < np.finfo(float).eps:
            axis = unitx
        else:
            axis = unitz

        mark = norm_vector(perpendicular_to_vectors(direction, axis)) * 0.03 * length
        mark_start1 = start + mark
        mark_start2 = start - mark
        mark_end1 = end + mark
        mark_end2 = end - mark
        ax.plot([mark_start1[0], mark_start2[0]],
                [mark_start1[1], mark_start2[1]],
                [mark_start1[2], mark_start2[2]],
                color=color)
        ax.plot([mark_end1[0], mark_end2[0]],
                [mark_end1[1], mark_end2[1]],
                [mark_end1[2], mark_end2[2]],
                color=color)
        text_location = start + 0.45 * direction
        if above:
            text_location[2] += 0.3 * length
        ax.text(text_location[0], text_location[1], text_location[2], name, zdir="x", **kwargs)

        return ax


    def plot_box(ax=None, size=np.ones(3), A2B=np.eye(4), ax_s=1, wireframe=True, color="k", alpha=1.0):
        """Plot box.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        size : array-like, shape (3,), optional (default: [1, 1, 1])
            Size of the box per dimension

        A2B : array-like, shape (4, 4)
            Center of the box

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe of cylinder and surface otherwise

        color : str, optional (default: black)
            Color in which the cylinder should be plotted

        alpha : float, optional (default: 1)
            Alpha value of the mesh that will be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        corners = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])
        corners = (corners - 0.5) * size
        corners = transform(
            A2B, np.hstack((corners, np.ones((len(corners), 1)))))[:, :3]

        if wireframe:
            for i, j in [(0, 1), (0, 2), (1, 3), (2, 3),
                         (4, 5), (4, 6), (5, 7), (6, 7),
                         (0, 4), (1, 5), (2, 6), (3, 7)]:
                ax.plot([corners[i, 0], corners[j, 0]],
                        [corners[i, 1], corners[j, 1]],
                        [corners[i, 2], corners[j, 2]],
                        c=color, alpha=alpha)
        else:
            p3c = Poly3DCollection(np.array([
                [corners[0], corners[1], corners[2]],
                [corners[1], corners[2], corners[3]],

                [corners[4], corners[5], corners[6]],
                [corners[5], corners[6], corners[7]],

                [corners[0], corners[1], corners[4]],
                [corners[1], corners[4], corners[5]],

                [corners[2], corners[6], corners[7]],
                [corners[2], corners[3], corners[7]],

                [corners[0], corners[4], corners[6]],
                [corners[0], corners[2], corners[6]],

                [corners[1], corners[5], corners[7]],
                [corners[1], corners[3], corners[7]],
            ]))
            p3c.set_alpha(alpha)
            p3c.set_facecolor(color)
            ax.add_collection3d(p3c)

        return ax


    def plot_sphere(ax=None, radius=1.0, p=np.zeros(3), ax_s=1, wireframe=True, n_steps=100, color="k", alpha=1.0):
        """Plot cylinder.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        radius : float, optional (default: 1)
            Radius of the sphere

        p : array-like, shape (3,), optional (default: [0, 0, 0])
            Center of the sphere

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe of cylinder and surface otherwise

        n_steps : int, optional (default: 100)
            Number of discrete steps plotted in each dimension

        color : str, optional (default: black)
            Color in which the cylinder should be plotted

        alpha : float, optional (default: 1)
            Alpha value of the mesh that will be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
        x = p[0] + radius * np.sin(phi) * np.cos(theta)
        y = p[1] + radius * np.sin(phi) * np.sin(theta)
        z = p[2] + radius * np.cos(phi)

        if wireframe:
            ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color=color, alpha=alpha)
        else:
            ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

        return ax


    def plot_cylinder(ax=None, length=1.0, radius=1.0, thickness=0.0, A2B=np.eye(4), ax_s=1, wireframe=True, n_steps=100, alpha=1.0, color="k"):
        """Plot cylinder.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        length : float, optional (default: 1)
            Length of the cylinder

        radius : float, optional (default: 1)
            Radius of the cylinder

        thickness : float, optional (default: 0)
            Thickness of a cylindrical shell. It will be subtracted from the
            outer radius to obtain the inner radius. The difference must be
            greater than 0.

        A2B : array-like, shape (4, 4)
            Center of the cylinder

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe of cylinder and surface otherwise

        n_steps : int, optional (default: 100)
            Number of discrete steps plotted in each dimension

        alpha : float, optional (default: 1)
            Alpha value of the mesh that will be plotted

        color : str, optional (default: black)
            Color in which the cylinder should be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        inner_radius = radius - thickness
        if inner_radius <= 0.0:
            raise ValueError("Thickness of cylindrical shell results in "
                             "invalid inner radius: %g" % inner_radius)

        axis_start = A2B.dot(np.array([0, 0, -0.5 * length, 1]))[:3]
        axis_end = A2B.dot(np.array([0, 0, 0.5 * length, 1]))[:3]
        axis = axis_end - axis_start
        axis /= length

        not_axis = np.array([1, 0, 0])
        if (axis == not_axis).all():
            not_axis = np.array([0, 1, 0])

        n1 = np.cross(axis, not_axis)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(axis, n1)

        if wireframe:
            t = np.linspace(0, length, n_steps)
        else:
            t = np.array([0, length])
        theta = np.linspace(0, 2 * np.pi, n_steps)
        t, theta = np.meshgrid(t, theta)

        if thickness > 0.0:
            X_outer, Y_outer, Z_outer = [
                axis_start[i] + axis[i] * t +
                radius * np.sin(theta) * n1[i] +
                radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            X_inner, Y_inner, Z_inner = [
                axis_end[i] - axis[i] * t +
                inner_radius * np.sin(theta) * n1[i] +
                inner_radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            X = np.hstack((X_outer, X_inner))
            Y = np.hstack((Y_outer, Y_inner))
            Z = np.hstack((Z_outer, Z_inner))
        else:
            X, Y, Z = [axis_start[i] + axis[i] * t +
                       radius * np.sin(theta) * n1[i] +
                       radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

        if wireframe:
            ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, alpha=alpha,
                              color=color)
        else:
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

        return ax


    def plot_mesh(ax=None, filename=None, A2B=np.eye(4), s=np.array([1.0, 1.0, 1.0]), ax_s=1, wireframe=False, convex_hull=False, alpha=1.0, color="k"):
        """Plot mesh.

        Note that this function requires the additional library 'trimesh'. It will
        print a warning if trimesh is not available.

        Parameters
        ----------
        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        filename : str, optional (default: None)
            Path to mesh file.

        A2B : array-like, shape (4, 4)
            Pose of the mesh

        s : array-like, shape (3,), optional (default: [1, 1, 1])
            Scaling of the mesh that will be drawn

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        wireframe : bool, optional (default: True)
            Plot wireframe of mesh and surface otherwise

        convex_hull : bool, optional (default: False)
            Show convex hull instead of the original mesh. This can be much
            faster.

        alpha : float, optional (default: 1)
            Alpha value of the mesh that will be plotted

        color : str, optional (default: black)
            Color in which the cylinder should be plotted

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if ax is None:
            ax = make_3d_axis(ax_s)

        if filename is None:
            warnings.warn(
                "Mesh will be ignored. You have to set a mesh path to "
                "plot meshes.")
            return ax

        try:
            import trimesh
        except ImportError:
            warnings.warn(
                "Cannot display mesh. Library 'trimesh' not installed.")
            return ax

        mesh = trimesh.load(filename)
        if convex_hull:
            mesh = mesh.convex_hull
        vertices = mesh.vertices * s
        vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
        vertices = transform(A2B, vertices)[:, :3]
        vectors = np.array([vertices[[i, j, k]] for i, j, k in mesh.faces])
        if wireframe:
            surface = Line3DCollection(vectors)
            surface.set_color(color)
        else:
            surface = Poly3DCollection(vectors)
            surface.set_facecolor(color)
        surface.set_alpha(alpha)
        ax.add_collection3d(surface)
        return ax


    def remove_frame(ax, left=0.0, bottom=0.0, right=1.0, top=1.0):
        """Remove axis and scale bbox.

        Parameters
        ----------
        ax : Matplotlib 3d axis
            Axis from which we remove the frame

        left : float, optional (default: 0)
            Position of left border (between 0 and 1)

        bottom : float, optional (default: 0)
            Position of bottom border (between 0 and 1)

        right : float, optional (default: 1)
            Position of right border (between 0 and 1)

        top : float, optional (default: 1)
            Position of top border (between 0 and 1)
        """
        ax.axis("off")
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)

except ImportError:
    warnings.warn("Matplotlib is not installed, visualization is not available")
