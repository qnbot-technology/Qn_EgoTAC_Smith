"""Verbatim function subset; provenance/extraction.json records origin."""
import numpy as np


def se3(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('Expected proper SE3: finite rigid transform, no scale/reflection')
    return matrix
