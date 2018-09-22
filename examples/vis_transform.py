import sys
import numpy as np
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QThread, pyqtSignal
import pyqtgraph.opengl as gl
import pytransform.rotations as pr
import pytransform.transformations as pt
# could be moved to library:
from OpenGL.GL import (
    glEnable, GL_LINE_SMOOTH, glHint, GL_LINE_SMOOTH_HINT, GL_NICEST, glBegin,
    GL_LINES, glColor4f, glVertex3f, glLineWidth, glEnd, GL_BLEND, glBlendFunc,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
from PyQt4.QtGui import QVector3D
from pyqtgraph.Transform3D import Transform3D
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class GLFrameItem(GLGraphicsItem):
    def __init__(self, A2B, s=1.0, lw=1.0, antialias=True,
                 glOptions='translucent'):
        GLGraphicsItem.__init__(self)

        self.A2B_ = A2B

        self.lw = lw
        self.antialias = antialias
        self.setSize(s, s, s)
        self.setGLOptions(glOptions)

    def setSize(self, x=None, y=None, z=None, size=None):
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        return self.__size[:]

    def setData(self, *args):
        self.A2B_ = np.reshape(args, (4, 4))
        self.update()

    def paint(self):
        R = self.A2B_[:3, :3]
        p = self.A2B_[:3, 3]

        self.setupGLState()
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(self.lw)
        glBegin(GL_LINES)

        size_x, size_y, size_z = self.size()
        for d in range(3):
            rgb = [0, 0, 0]
            rgb[d] = 1
            glColor4f(rgb[0], rgb[1], rgb[2], 1)
            glVertex3f(p[0], p[1], p[2])
            glVertex3f(p[0] + size_x * R[0, d],
                       p[1] + size_y * R[1, d],
                       p[2] + size_z * R[2, d])
        glEnd()


class GLRefItem(GLGraphicsItem):
    def __init__(self, transform=None, size=None, antialias=True, glOptions='translucent'):
        GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if transform is not None:
            self.setTransform(Transform3D(*np.ravel(transform).tolist()))
        if size is None:
            size = QVector3D(20, 20, 1)
        self.setSize(size=size)
        self.setSpacing(1, 1, 1)

    def setSize(self, x=None, y=None, z=None, size=None):
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        return self.__size[:]

    def setSpacing(self, x=None, y=None, z=None, spacing=None):
        if spacing is not None:
            x = spacing.x()
            y = spacing.y()
            z = spacing.z()
        self.__spacing = [x, y, z]
        self.update()

    def spacing(self):
        return self.__spacing[:]

    def paint(self):
        self.setupGLState()

        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glBegin(GL_LINES)

        x, y, z = self.size()
        xs, ys, zs = self.spacing()
        xvals = np.arange(-x / 2., x / 2. + xs * 0.001, xs)
        yvals = np.arange(-y / 2., y / 2. + ys * 0.001, ys)
        glColor4f(0, 0, 0, 0.1)
        for x in xvals:
            glVertex3f(x, yvals[0], 0)
            glVertex3f(x, yvals[-1], 0)
        for y in yvals:
            glVertex3f(xvals[0], y, 0)
            glVertex3f(xvals[-1], y, 0)

        glEnd()


class AnimationThread(QThread):
    # for each provided transformation we define a new signal
    new_A = pyqtSignal(
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float)
    new_B = pyqtSignal(
        float, float, float, float,
        float, float, float, float,
        float, float, float, float,
        float, float, float, float)

    def __init__(self):
        super(AnimationThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        i = 1.0
        cycle_length = 1000.0
        while True:
            if i % cycle_length == 0.0:
                self.msleep(200)
                i = 0.0
            t = i / cycle_length * np.pi * 2.0

            A = pt.transform_from(
                R=pr.matrix_from_angle(0, t),
                p=np.array([-5.0 * np.sin(t), 5.0 * np.cos(t), 5.0 * np.sin(t)])
            )
            self.new_A.emit(*np.ravel(A).tolist())

            B = pt.transform_from(
                R=pr.matrix_from_angle(1, t),
                p=np.array([5.0 * np.cos(t), -5.0 * np.sin(t), 5.0 * np.sin(t)])
            )
            self.new_B.emit(*np.ravel(B).tolist())

            i += 1.0


app = QApplication(sys.argv)

view = gl.GLViewWidget()
view.setCameraPosition(distance=50)
view.setBackgroundColor(255, 255, 255)
view.setWindowTitle("Example")
view.setFixedSize(800, 600)
view.show()

# add grids
view.addItem(GLRefItem())
view.addItem(GLRefItem(pt.transform_from(
    R=pr.matrix_from_angle(0, np.pi / 2.0), p=np.zeros(3))))
view.addItem(GLRefItem(pt.transform_from(
    R=pr.matrix_from_angle(1, np.pi / 2.0), p=np.zeros(3))))

# add base coordinate system
origin = GLFrameItem(np.eye(4), s=10, lw=10.0)
view.addItem(origin)

# add frames that will be animated
animated_frame_A = GLFrameItem(np.eye(4), s=3, lw=3)
view.addItem(animated_frame_A)
animated_frame_B = GLFrameItem(np.eye(4), s=3, lw=3)
view.addItem(animated_frame_B)

worker = AnimationThread()
# here we connect producer and consumers
worker.new_A.connect(animated_frame_A.setData)
worker.new_B.connect(animated_frame_B.setData)
worker.start()

app.exec_()
