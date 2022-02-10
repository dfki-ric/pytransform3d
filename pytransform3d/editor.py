"""Modify transformations visually."""
qt_available = False
qt_version = None
try:
    import PyQt5.QtCore as QtCore
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QSlider, QDoubleSpinBox,
        QGridLayout, QLabel, QGroupBox, QHBoxLayout, QComboBox, QVBoxLayout)
    qt_available = True
    qt_version = 5
except ImportError:
    try:
        import PyQt4.QtCore as QtCore
        from PyQt4.QtGui import (
            QApplication, QMainWindow, QWidget, QSlider, QDoubleSpinBox,
            QGridLayout, QLabel, QGroupBox, QHBoxLayout, QComboBox,
            QVBoxLayout)
        qt_available = True
        qt_version = 4
    except ImportError:
        import warnings
        warnings.warn(
            "Cannot import PyQt. TransformEditor won't be available.")
        TransformEditor = None


if qt_available:
    import sys
    from functools import partial
    from contextlib import contextmanager
    import numpy as np
    from .transform_manager import TransformManager
    from .rotations import (active_matrix_from_intrinsic_euler_xyz,
                            intrinsic_euler_xyz_from_active_matrix)
    from .transformations import transform_from
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.figure import Figure
    if qt_version == 5:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
    else:
        assert qt_version == 4
        from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT

    @contextmanager
    def _block_signals(qobject):
        """Block signals of a QObject in this context."""
        signals_blocked = qobject.blockSignals(True)
        try:
            yield qobject
        finally:
            qobject.blockSignals(signals_blocked)


    def _internal_repr(A2B):
        """Compute internal representation of transform."""
        p = A2B[:3, 3]
        R = A2B[:3, :3]
        e = intrinsic_euler_xyz_from_active_matrix(R)
        return np.hstack((p, e))


    class PositionEulerEditor(QWidget):
        """Frame editor that represents orientation by Euler angles (XY'Z'').

        Parameters
        ----------
        base_frame : string
            Name of the base frame

        xlim : tuple
            Lower and upper limit for the x position. Defines the range of the
            plot and the range of the slider.

        ylim : tuple
            Lower and upper limit for the y position. Defines the range of the
            plot and the range of the slider.

        zlim : tuple
            Lower and upper limit for the z position. Defines the range of the
            plot and the range of the slider.

        parent : QWidget, optional (default: None)
            Parent widget.
        """
        frameChanged = QtCore.pyqtSignal()

        def __init__(self, base_frame, xlim, ylim, zlim, parent=None):
            super(PositionEulerEditor, self).__init__(parent)

            self.dim_labels = ["x", "y", "z", "X", "Y'", "Z''"]
            self.limits = [xlim, ylim, zlim,
                           (-3.141, 3.141), (-1.570, 1.570), (-3.141, 3.141)]
            self.n_slider_steps = [int(100 * (upper - lower)) + 1
                                   for lower, upper in self.limits]
            self.setLayout(self._create(base_frame))
            self.A2B = None

        def _create(self, base_frame):
            self.sliders = []
            self.spinboxes = []
            for i in range(len(self.dim_labels)):
                self.sliders.append(QSlider(QtCore.Qt.Horizontal))
                self.sliders[i].setRange(0, self.n_slider_steps[i])
                self.sliders[i].valueChanged.connect(
                    partial(self._on_slide, i))
                spinbox = QDoubleSpinBox()
                spinbox.setRange(*self.limits[i])
                spinbox.setDecimals(3)
                spinbox.setSingleStep(0.001)
                self.spinboxes.append(spinbox)
                self.spinboxes[i].valueChanged.connect(
                    partial(self._on_pos_edited, i))
            slider_group = QGridLayout()
            slider_group.addWidget(QLabel("Position"),
                                   0, 0, 1, 3, QtCore.Qt.AlignCenter)
            slider_group.addWidget(QLabel("Orientation (Euler angles)"),
                                   0, 3, 1, 3, QtCore.Qt.AlignCenter)
            for i, slider in enumerate(self.sliders):
                slider_group.addWidget(QLabel(self.dim_labels[i]), 1, i)
                slider_group.addWidget(slider, 2, i)
                slider_group.addWidget(self.spinboxes[i], 3, i)
            slider_groupbox = QGroupBox("Transformation in frame '%s'"
                                        % base_frame)
            slider_groupbox.setLayout(slider_group)
            layout = QHBoxLayout()
            layout.addWidget(slider_groupbox)
            layout.addStretch(1)
            return layout

        def set_frame(self, A2B):
            """Set pose of frame.

            Parameters
            ----------
            A2B : array
                Transformation matrix
            """
            self.A2B = A2B
            pose = _internal_repr(self.A2B)

            for i in range(6):
                pos = self._pos_to_slider_pos(i, pose[i])
                with _block_signals(self.sliders[i]) as slider:
                    slider.setValue(pos)
                with _block_signals(self.spinboxes[i]) as spinbox:
                    spinbox.setValue(pose[i])

        def _on_pos_edited(self, dim, pos):
            """Slot: value in spinbox changed."""
            pose = _internal_repr(self.A2B)
            pose[dim] = pos
            self.A2B = transform_from(active_matrix_from_intrinsic_euler_xyz(
                pose[3:]), pose[:3])

            for i in range(6):
                pos = self._pos_to_slider_pos(i, pose[i])
                with _block_signals(self.sliders[i]) as slider:
                    slider.setValue(pos)

            self.frameChanged.emit()

        def _on_slide(self, dim, step):
            """Slot: slider position changed."""
            pose = _internal_repr(self.A2B)
            v = self._slider_pos_to_pos(dim, step)
            pose[dim] = v
            self.A2B = transform_from(active_matrix_from_intrinsic_euler_xyz(
                pose[3:]), pose[:3])

            self.spinboxes[dim].setValue(v)

            self.frameChanged.emit()

        def _pos_to_slider_pos(self, dim, pos):
            """Compute slider position from value."""
            m = self.limits[dim][0]
            r = self.limits[dim][1] - m
            slider_pos = int((pos - m) / r * self.n_slider_steps[dim])
            slider_pos = np.clip(slider_pos, 0, self.n_slider_steps[dim])
            return slider_pos

        def _slider_pos_to_pos(self, dim, slider_pos):
            """Create value from slider position."""
            m = self.limits[dim][0]
            r = self.limits[dim][1] - m
            return m + r * float(slider_pos) / float(self.n_slider_steps[dim])

    class TransformEditor(QMainWindow):
        """GUI to edit transformations.

        .. warning::

            Note that this module requires PyQt4.

        Parameters
        ----------
        transform_manager : TransformManager
            All nodes that are reachable from the base frame will be editable

        frame : string
            Name of the base frame

        xlim : tuple, optional (-1, 1)
            Lower and upper limit for the x position. Defines the range of the
            plot and the range of the slider.

        ylim : tuple, optional (-1, 1)
            Lower and upper limit for the y position. Defines the range of the
            plot and the range of the slider.

        zlim : tuple, optional (-1, 1)
            Lower and upper limit for the z position. Defines the range of the
            plot and the range of the slider.

        s : float, optional (default: 1)
            Scaling of the axis and angle that will be drawn

        figsize : tuple of integers, optional (default: (10, 10))
            Width, height in inches.

        dpi : integer, optional (default: 100)
            Resolution of the figure.

        parent : QWidget, optional (default: None)
            Parent widget.

        Attributes
        ----------
        transform_manager : TransformManager
            Result, all frames are expressed in the base frame
        """
        def __init__(self, transform_manager, base_frame, xlim=(-1.0, 1.0),
                     ylim=(-1.0, 1.0), zlim=(-1.0, 1.0), s=1.0,
                     figsize=(10, 10), dpi=100, window_size=(500, 600),
                     parent=None):
            self.app = QApplication(sys.argv)

            super(TransformEditor, self).__init__(parent)
            self.transform_manager = self._init_transform_manager(
                transform_manager, base_frame)
            self.base_frame = base_frame
            self.xlim = xlim
            self.ylim = ylim
            self.zlim = zlim
            self.s = s
            self.figsize = figsize
            self.dpi = dpi
            self.window_size = window_size

            self.setWindowTitle("Transformation Editor")
            self.canvas = None
            self.fig = None
            self.axis = None

            self._create_main_frame()
            self._on_node_changed(0)

        def _init_transform_manager(self, transform_manager, frame):
            """Transform all nodes into the reference frame."""
            tm = TransformManager()
            if frame not in transform_manager.nodes:
                raise KeyError("Unknown frame '%s'" % frame)

            for node in transform_manager.nodes:
                try:
                    node2frame = transform_manager.get_transform(node, frame)
                    tm.add_transform(node, frame, node2frame)
                except KeyError:
                    pass  # Frame is not connected to the reference frame

            return tm

        def _create_main_frame(self):
            """Create main frame and layout."""
            self.main_frame = QWidget()

            self.frame_editor = PositionEulerEditor(
                self.base_frame, self.xlim, self.ylim, self.zlim)
            self.frame_editor.frameChanged.connect(self._on_update)

            self.frame_selection = self._create_frame_selector()

            plot = self._create_plot()

            vbox = QVBoxLayout()
            vbox.addWidget(self.frame_editor)
            vbox.addWidget(self.frame_selection)
            vbox.addWidget(plot)

            main_layout = QHBoxLayout()
            main_layout.addLayout(vbox)

            self.main_frame.setLayout(main_layout)
            self.setCentralWidget(self.main_frame)
            self.setGeometry(0, 0, *self.window_size)

        def _create_frame_selector(self):
            frame_selection = QComboBox()
            for node in self.transform_manager.nodes:
                if node != self.base_frame:
                    frame_selection.addItem(node)
            frame_selection.activated.connect(self._on_node_changed)
            return frame_selection

        def _create_plot(self):
            plot = QWidget()
            canvas_group = QGridLayout()
            self.fig = Figure(self.figsize, dpi=self.dpi)
            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setParent(self.main_frame)
            canvas_group.addWidget(self.canvas, 1, 0)
            mpl_toolbar = NavigationToolbar2QT(self.canvas, self.main_frame)
            canvas_group.addWidget(mpl_toolbar, 2, 0)
            plot.setLayout(canvas_group)
            return plot

        def _on_node_changed(self, node_idx):
            """Slot: manipulatable node changed."""
            self.node = self.frame_selection.itemText(node_idx)
            A2B = self.transform_manager.get_transform(
                self.node, self.base_frame)
            self.frame_editor.set_frame(A2B)
            self._plot()

        def _on_update(self):
            """Slot: transformation changed."""
            self.transform_manager.add_transform(
                self.node, self.base_frame, self.frame_editor.A2B)
            self._plot()

        def _plot(self):
            """Draw plot."""
            if self.axis is None:
                elev, azim = 30, 60
            else:
                elev, azim = self.axis.elev, self.axis.azim
                self.fig.delaxes(self.axis)

            self.axis = self.fig.add_subplot(111, projection="3d")
            self.axis.view_init(elev, azim)

            self.axis.set_xlim(self.xlim)
            self.axis.set_ylim(self.ylim)
            self.axis.set_zlim(self.zlim)

            p = self.transform_manager.get_transform(
                self.node, self.base_frame)[:3, 3]
            self.axis.scatter(p[0], p[1], p[2], s=100)
            self.transform_manager.plot_frames_in(
                self.base_frame, ax=self.axis, s=self.s)

            self.canvas.draw()

        def show(self):
            """Start GUI."""
            super(TransformEditor, self).show()
            self.app.exec_()
