import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as RotFunc


def scale_to_unit_sphere(points, center=None, buffer=1.0):
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    #     midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points**2, axis=1))) * buffer
    points = points / scale
    return points


def scale_to_unit_cube(vertices, ret_scale=False):
    # normalize diagonal=1
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale
    if ret_scale:
        return vertices, scale
    return vertices


def zero_center_pts(pts):
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts


def compute_cotangent(vec1, vec2):
    # Check for 2D vectors and compute directionality using cross product
    if len(vec1) == 2 and len(vec2) == 2:
        sign = np.sign(np.cross(vec1, vec2))
    else:  # In 3D or other dimensions, we'll just assume positive
        sign = 1

    norm_vec1 = vec1 / np.linalg.norm(vec1)
    norm_vec2 = vec2 / np.linalg.norm(vec2)

    cosine_angle = np.dot(norm_vec1, norm_vec2)

    angle_rad = sign * np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    if np.isclose(angle_rad, np.pi / 2, atol=1e-10) or np.isclose(
        angle_rad, -np.pi / 2, atol=1e-10
    ):
        return 0.0
    else:
        cotangent = 1 / np.tan(angle_rad)
        return cotangent


def vec_normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)


def Mat_F_to_C_sp(mat_sp):
    # Given in F order, reshape to C order
    # XXX, YYY, ZZZ --> XYZ, XYZ, XYZ
    mat_sp_x = mat_sp[: mat_sp.shape[0] // 3, :]
    mat_sp_y = mat_sp[mat_sp.shape[0] // 3 : 2 * mat_sp.shape[0] // 3, :]
    mat_sp_z = mat_sp[2 * mat_sp.shape[0] // 3 :, :]
    mat_sp_c = sp.hstack([mat_sp_x, mat_sp_y, mat_sp_z]).reshape(-1, mat_sp.shape[1])
    return mat_sp_c


def get_eye_like_np(mat):
    # mat: identity matrix of shape (n, m, m)
    return np.eye(mat.shape[1])[np.newaxis, :, :].repeat(mat.shape[0], axis=0)


def polar_decomp_np(my_mat):
    # Polar decomposition in terms of singular-value decomposition
    if my_mat.ndim == 2:
        U, S, Vh = np.linalg.svd(my_mat)
        u = U @ Vh
        p = Vh.T.conj() @ np.diag(S) @ Vh
    elif my_mat.ndim == 3:
        U, S, Vh = np.linalg.svd(my_mat)
        u = U @ Vh
        diag_S = get_eye_like_np(S) * S[:, np.newaxis, :]
        p = Vh.transpose(0, 2, 1).conj() @ diag_S @ Vh
    else:
        raise ValueError("Input tensor must be 2 or 3 dimensional")
    return u, p


def get_rotation_to_plane(source_normals, common_normal):
    """
    Get the rotation matrix to align the normal to the z-axis
    """
    # Get the rotation matrix
    rot_axis = np.cross(source_normals, common_normal[None, :])
    rot_angle = np.arccos(np.einsum("ij,ij->i", source_normals, common_normal[None, :]))
    rot_norm = np.linalg.norm(rot_axis, axis=1, keepdims=True)
    rot_norm[rot_norm == 0] = 1
    rot_axis_norm = rot_axis / rot_norm
    rot_mat_src = RotFunc.from_rotvec(rot_axis_norm * rot_angle[:, None]).as_matrix()
    # special case when rot_axis is 0
    degenerate = np.linalg.norm(rot_axis, axis=1) < 1e-6
    rot_mat_src[degenerate] = np.eye(3)
    return rot_mat_src


def blockify(my_vec):
    """
    Convert a vector of shape (n_v, 3) to a block diagonal matrix of shape (n_v, n_v*3)
    Args:
        my_vec (np.array): (n_v, 3) vector
    Returns:
        csr_matrix: (n_v, n_v*3) block diagonal matrix
    """
    n_vb = my_vec.shape[0]
    row_idx = np.arange(n_vb).repeat(3)
    col_idx = np.tile(np.array([0, 1, 2]), n_vb) + row_idx * 3
    return sp.csr_matrix((my_vec.flatten(), (row_idx, col_idx)), shape=(n_vb, n_vb * 3))


def create_rep_mat(n_v, k):
    """
    Create a matrix that repeats the incidence matrix k times
    ** To be multiplied to right of `blockify` output.**
    [[1, 1, 1, 0, 0 ....],
     [0, 0, 0, 1, 1, 1 ....],]
    Args:
        n_v (int): number of vertices
        k (int): repetitions

    Returns:
        _type_: _description_
    """
    n_vb = n_v * k
    inc_mat_col = np.arange(n_v * 3).reshape(-1, 3).repeat(k, axis=0).flatten()  # n_v*3
    inc_mat_row = np.arange(n_vb * 3)  # n_vb*3
    inc_mat_data = np.ones(inc_mat_col.shape[0])  # n_vb*3
    inc_mat = sp.csr_matrix(
        (inc_mat_data, (inc_mat_row, inc_mat_col)), shape=(n_vb * 3, n_v * 3)
    )
    return inc_mat


def blockify_transpose(my_vec):
    """
    [[v1, v2, v3, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, v1, v2, v3, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, v1, v2, v3]]

     **NOTE**: `blockify()@vec` will give transpose of
     `blockify_transpose()@vec`.

     multiplying from right by flattened 3x3 will give
      `Mat @ col_vec`

    Args:
        my_vec (np.array): (n_v, 3) vector
    Returns:
        csr_matrix: (n_v*3, n_v*9) block diagonal matrix
    """
    n_vb = my_vec.shape[0]
    row_idx = np.arange(n_vb * 3).repeat(3)
    col_idx_shift = np.arange(n_vb).repeat(9) * 9  # 3x3 matrix
    col_idx_offset = np.tile(
        (np.arange(3) + np.array([0, 3, 6])[:, None]).flatten(), n_vb
    )
    col_idx = col_idx_shift + col_idx_offset
    data = my_vec.repeat(3, axis=0).flatten()
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_vb * 3, n_vb * 9))
