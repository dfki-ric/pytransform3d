"""
=====================
Transformation Editor
=====================

The transformation editor can be used to manipulate transformations.
"""
from pytransform3d.transform_manager import TransformManager
from pytransform3d.editor import TransformEditor
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import active_matrix_from_extrinsic_euler_xyx


tm = TransformManager()

tm.add_transform(
    "tree", "world",
    transform_from(
        active_matrix_from_extrinsic_euler_xyx([0, 0.5, 0]),
        [0, 0, 0.5]
    )
)
tm.add_transform(
    "car", "world",
    transform_from(
        active_matrix_from_extrinsic_euler_xyx([0.5, 0, 0]),
        [0.5, 0, 0]
    )
)

te = TransformEditor(tm, "world", s=0.3)
te.show()
print("tree to world:")
print(te.transform_manager.get_transform("tree", "world"))
