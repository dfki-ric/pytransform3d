import sys
from functools import partial
try:
    import PyQt4.QtCore as QtCore
    import PyQt4.QtGui as QtGui
except ImportError as e:
    print("Please install PyQt4")
    raise e
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .transform_manager import TransformManager
from .rotations import matrix_from_euler_xyz, euler_xyz_from_matrix
from .transformations import transform_from


class TransformEditor(QtGui.QMainWindow):
    """GUI to edit transformations.

    Parameters
    ----------
    transform_manager : TransformManager
        All nodes that are reachable from the base frame will be editable

    frame : string
        Name of the base frame

    xlim : tuple, optional (-1, 1)
        Lower and upper limit for the x position. Defines the range of the plot
        and the range of the slider.

    ylim : tuple, optional (-1, 1)
        Lower and upper limit for the y position. Defines the range of the plot
        and the range of the slider.

    zlim : tuple, optional (-1, 1)
        Lower and upper limit for the z position. Defines the range of the plot
        and the range of the slider.

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    figsize : tuple of integers, optional (default: (10, 10))
        Width, height in inches.

    dpi : integer, optional (default: 100)
        Resolution of the figure.

    parent : QtGui.QWidget, optional (default: None)
        Parent widget.

    Attributes
    ----------
    transform_manager : TransformManager
        Result, all frames are expressed in the base frame
    """
    def __init__(self, transform_manager, frame, xlim=(-1.0, 1.0),
                 ylim=(-1.0, 1.0), zlim=(-1.0, 1.0), s=1.0, figsize=(10, 10),
                 dpi=100, parent=None):
        self.app = QtGui.QApplication(sys.argv)

        super(TransformEditor, self).__init__(parent)
        self.transform_manager = self._init_transform_manager(
            transform_manager, frame)
        self.frame = frame
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.s = s
        self.figsize = figsize
        self.dpi = dpi

        self.slider_limits = [xlim, ylim, zlim,
                              (-np.pi, np.pi), (-0.5 * np.pi, 0.5 * np.pi),
                              (-np.pi, np.pi)]

        self.n_slider_steps = 1000

        self.setWindowTitle("Transformation Editor")
        self.dim_labels = ["x", "y", "z", "X", "Y'", "Z''"]
        self.axis = None

        self._create_main_frame()
        self._on_node_changed([node for node in self.transform_manager.nodes
                               if node != self.frame][0])

    def _init_transform_manager(self, transform_manager, frame):
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
        self.main_frame = QtGui.QWidget()

        plot = QtGui.QWidget()
        canvas_group = QtGui.QGridLayout()
        self.fig = Figure(self.figsize, dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)
        canvas_group.addWidget(self.canvas, 1, 0)
        mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)
        canvas_group.addWidget(mpl_toolbar, 2, 0)
        plot.setLayout(canvas_group)

        self.sliders = []
        self.spinboxes = []
        for i in range(len(self.dim_labels)):
            self.sliders.append(QtGui.QSlider(QtCore.Qt.Horizontal))
            self.sliders[i].setRange(0, self.n_slider_steps)
            self.connect(self.sliders[i],
                         QtCore.SIGNAL("valueChanged(int)"),
                         partial(self._on_slide, i))
            spinbox = QtGui.QDoubleSpinBox()
            spinbox.setRange(*self.slider_limits[i])
            spinbox.setDecimals(4)
            spinbox.setSingleStep(0.0001)
            self.spinboxes.append(spinbox)
            self.connect(self.spinboxes[i],
                         QtCore.SIGNAL("valueChanged(double)"),
                         partial(self._on_edited, i))

        slider_group = QtGui.QGridLayout()
        slider_group.addWidget(QtGui.QLabel("Position"),
                               0, 0, 1, 3, QtCore.Qt.AlignCenter)
        slider_group.addWidget(QtGui.QLabel("Orientation (Euler angles)"),
                               0, 3, 1, 3, QtCore.Qt.AlignCenter)
        for i, slider in enumerate(self.sliders):
            slider_group.addWidget(QtGui.QLabel(self.dim_labels[i]), 1, i)
            slider_group.addWidget(slider, 2, i)
            slider_group.addWidget(self.spinboxes[i], 3, i)
        slider_groupbox = QtGui.QGroupBox("Transformation in frame '%s'"
                                          % self.frame)
        slider_groupbox.setLayout(slider_group)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(slider_groupbox)
        hbox.addStretch(1)

        frame_selection = QtGui.QComboBox()
        for node in self.transform_manager.nodes:
            if node != self.frame:
                frame_selection.addItem(node)
        self.connect(frame_selection, QtCore.SIGNAL("activated(const QString&)"),
                     self._on_node_changed)

        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(frame_selection)
        vbox.addWidget(plot)

        main_layout = QtGui.QHBoxLayout()
        main_layout.addLayout(vbox)

        self.main_frame.setLayout(main_layout)
        self.setCentralWidget(self.main_frame)
        self.setGeometry(0, 0, 500, 600)

    def _on_node_changed(self, node):
        self.node = str(node)
        pose = self._get_pose()

        for i in range(6):
            pos = self._pos_to_slider_pos(i, pose[i])
            self.sliders[i].setValue(pos)
            self.spinboxes[i].setValue(pose[i])

        self._plot()

    def _on_edited(self, dim, pos):
        pose = self._get_pose()
        pose[dim] = pos
        for i in range(6):
            pos = self._pos_to_slider_pos(i, pose[i])
            self.sliders[i].setValue(pos)

    def _on_slide(self, dim, step):
        pose = self._get_pose()
        v = self._slider_pos_to_pos(dim, step)
        pose[dim] = v
        node2frame = transform_from(matrix_from_euler_xyz(pose[3:]), pose[:3])
        self.transform_manager.add_transform(self.node, self.frame, node2frame)
        self.spinboxes[dim].setValue(v)

        self._plot()

    def _get_pose(self):
        node2frame = self.transform_manager.get_transform(
            self.node, self.frame)
        p = node2frame[:3, 3]
        R = node2frame[:3, :3]
        e = euler_xyz_from_matrix(R)
        return np.hstack((p, e))

    def _pos_to_slider_pos(self, dim, pos):
        m = self.slider_limits[dim][0]
        r = self.slider_limits[dim][1] - m
        slider_pos = int((pos - m) / r * self.n_slider_steps)
        slider_pos = np.clip(slider_pos, 0, self.n_slider_steps)
        return slider_pos

    def _slider_pos_to_pos(self, dim, slider_pos):
        m = self.slider_limits[dim][0]
        r = self.slider_limits[dim][1] - m
        return m + r * float(slider_pos) / float(self.n_slider_steps)

    def _plot(self):
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

        p = self.transform_manager.get_transform(self.node, self.frame)[:3, 3]
        self.axis.scatter(p[0], p[1], p[2], s=100)
        self.transform_manager.plot_frames_in(
            self.frame, ax=self.axis, s=self.s)

        self.canvas.draw()

    def show(self):
        super(TransformEditor, self).show()
        self.app.exec_()
