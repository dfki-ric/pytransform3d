import sys
import numpy as np
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import QThread
import pyqtgraph.opengl as gl
import pytransform.rotations as pr
import pytransform.transformations as pt


from OpenGL.GL import (
    glEnable, GL_LINE_SMOOTH, glHint, GL_LINE_SMOOTH_HINT,
    GL_NICEST, glBegin, GL_LINES, glColor4f, glVertex3f, glLineWidth, glEnd)
class GLFrameItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, A2B, s=1.0, lw=1.0, antialias=True,
                 glOptions='translucent'):
        super(GLFrameItem, self).__init__()

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

    def set_data(self, A2B):
        self.A2B_ = A2B
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


from OpenGL.GL import (GL_BLEND, glBlendFunc, GL_SRC_ALPHA,
                       GL_ONE_MINUS_SRC_ALPHA)
from PyQt4.QtGui import QVector3D
class GLRefItem(gl.GLGraphicsItem.GLGraphicsItem):
    def __init__(self, size=None, color=None, antialias=True,
                 glOptions='translucent'):
        gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
        self.setGLOptions(glOptions)
        self.antialias = antialias
        if size is None:
            size = QVector3D(20, 20, 1)
        self.setSize(size=size)
        self.setSpacing(1, 1, 1)

    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x, y, z]
        self.update()

    def size(self):
        return self.__size[:]

    def setSpacing(self, x=None, y=None, z=None, spacing=None):
        """
        Set the spacing between grid lines.
        Arguments can be x,y,z or spacing=QVector3D().
        """
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


class Worker(QThread):
    def __init__(self, view):
        super(Worker, self).__init__()
        self.view = view

    def __del__(self):
        self.wait()

    def run(self):
        i = 1.0
        frame = GLFrameItem(A2B=np.eye(4))
        self.view.addItem(frame)
        while True:
            self.msleep(1)
            t = i / 1000.0
            A2B = pt.transform_from(
                R=pr.matrix_from_euler_xyz([t * np.pi * 2.0, 0, 0]),
                p=np.array([5, 5, 5.0 * np.sin(t)])
            )
            frame.set_data(A2B)
            i += 1.0


app = QApplication(sys.argv)

view = gl.GLViewWidget()
view.opts["distance"] = 20
view.setBackgroundColor(255, 255, 255)
view.setWindowTitle("Example")
view.setFixedSize(800, 600)
view.show()

grid = GLRefItem()
view.addItem(grid)

from pyqtgraph.Transform3D import Transform3D
grid = GLRefItem()
grid_origin = pt.transform_from(
    R=pr.matrix_from_angle(0, np.pi / 2.0), p=np.zeros(3))
grid.setTransform(Transform3D(*np.ravel(grid_origin).tolist()))
view.addItem(grid)

grid = GLRefItem()
grid_origin = pt.transform_from(
    R=pr.matrix_from_angle(1, np.pi / 2.0), p=np.zeros(3))
grid.setTransform(Transform3D(*np.ravel(grid_origin).tolist()))
view.addItem(grid)

origin = GLFrameItem(np.eye(4), s=10, lw=5.0)
view.addItem(origin)

worker = Worker(view)
worker.start()

app.exec_()
