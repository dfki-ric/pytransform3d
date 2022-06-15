"""Constants of rotation module."""
import numpy as np


unitx = np.array([1.0, 0.0, 0.0])
unity = np.array([0.0, 1.0, 0.0])
unitz = np.array([0.0, 0.0, 1.0])

R_id = np.eye(3)
a_id = np.array([1.0, 0.0, 0.0, 0.0])
q_id = np.array([1.0, 0.0, 0.0, 0.0])
q_i = np.array([0.0, 1.0, 0.0, 0.0])
q_j = np.array([0.0, 0.0, 1.0, 0.0])
q_k = np.array([0.0, 0.0, 0.0, 1.0])
p0 = np.array([0.0, 0.0, 0.0])

eps = 1e-7
