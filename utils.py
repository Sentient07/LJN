import scipy.sparse.linalg as sla
import hashlib
from glob import glob
from itertools import combinations, product
from cholespy import CholeskySolverD, MatrixType
from collections import defaultdict
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.sparse import csc_matrix, diags, coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree as KDTree
from torch_sparse.tensor import SparseTensor
from trimesh.transformations import rotation_matrix
import igl
import meshplot as mp
import numpy as np
import os
import os.path as osp
import potpourri3d as pp3d
import robust_laplacian
import random
import scipy
import scipy.sparse as sp
import sklearn.neighbors
import torch
from torch.linalg import norm as th_norm
import trimesh
from MyMesh import MyMesh


def vec_normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8)


def vec_normalize_th(vec):
    return vec / (th_norm(vec, dim=-1, keepdim=True) + 1e-8)


def numpied(th_tensor):
    if isinstance(th_tensor, np.ndarray):
        return th_tensor
    elif isinstance(th_tensor, torch.Tensor):
        return np.ascontiguousarray(th_tensor.detach().cpu().numpy())
    else:
        raise ValueError("Unknown format")


def torched(np_array, device="cuda", dtype=torch.float32):
    if isinstance(np_array, torch.Tensor):
        return np_array
    if isinstance(np_array, list):
        np_array = np.array(np_array)

    if np_array.dtype == "O":
        raise ValueError("Shouldn't be Object")
    return torch.from_numpy(np_array).to(device).to(dtype)


def f_from_m(m):
    m = trimesh.load(m, process=False)
    if isinstance(m, trimesh.Scene):
        return np.array(m.dump().sum().faces)
    return np.array(m.faces)


def check_is_trimesh(t_obj):
    if isinstance(t_obj, trimesh.Trimesh) or isinstance(
        t_obj, trimesh.points.PointCloud
    ):
        return True


def v_from_m(m):
    if isinstance(m, str):
        return np.array(trimesh.load(m, process=False).vertices)
    elif isinstance(m, np.ndarray):
        # assume it's vert already
        return m
    elif check_is_trimesh(m):
        return np.array(m.vertices)
    elif isinstance(m, trimesh.Scene):
        return np.array(m.dump().sum().vertices)
    else:
        raise ValueError("Unknown format")


def trimesh_from_vf(v, f):
    verts = numpied(v)
    faces = numpied(f).astype(np.int32)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def load_m(m):
    if isinstance(m, trimesh.Trimesh):
        return m
    loaded_obj = trimesh.load(m, process=False)
    if isinstance(loaded_obj, trimesh.Scene):
        loaded_obj = loaded_obj.dump().sum()
    return loaded_obj


def scale_to_unit_sphere(points, center=None, buffer=1.0):
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    #     midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points**2, axis=1))) * buffer
    points = points / scale
    return points


def scale_m_to_unit_sphere(mesh, buffer=1.0):
    m = load_m(mesh)
    return trimesh_from_vf(
        scale_to_unit_sphere(v_from_m(m), buffer=buffer), f_from_m(m)
    )


def fn_from_m(mesh):
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, process=False)
    return mesh.face_normals


def get_face_mass_matrix(vertices, faces, is_sparse):
    # d_area = igl.doublearea(vertices,faces)
    d_area = trimesh.triangles.area(vertices[faces])
    d_area = np.hstack((d_area, d_area, d_area))
    if is_sparse:
        return csc_matrix(diags(d_area))
    return diags(d_area)


def get_diverging_quantity(v, f, vector_fields):
    source_grad = igl.grad(v, f).astype(np.float32)
    mass_mat_3F = get_face_mass_matrix(v, f, is_sparse=True)
    source_adj = source_grad.T @ mass_mat_3F
    return source_adj @ vector_fields


def scipy_CD_idx(sA, sB):
    treeA = KDTree(sA)
    treeB = KDTree(sB)
    distB2A, a_nn_idx = treeA.query(sB)
    distA2B, b_nn_idx = treeB.query(sA)
    return distB2A, distA2B, a_nn_idx, b_nn_idx


def plot_CD_error(recon_m, gt_m, source_m, npoints=100000):
    import torch_scatter

    recon_m = scale_m_to_unit_sphere(recon_m)
    gt_m = scale_m_to_unit_sphere(gt_m)
    source_m = scale_m_to_unit_sphere(source_m)
    recon_samp, recon_f_idx = trimesh.sample.sample_surface_even(recon_m, npoints)
    gt_samp, gt_f_idx = trimesh.sample.sample_surface_even(gt_m, npoints)
    cd_gt_recon, cd_recon_gt, _, _ = scipy_CD_idx(recon_samp, gt_samp)
    cd_error = torch_scatter.scatter_mean(
        torch.from_numpy(cd_recon_gt), torch.from_numpy(recon_f_idx), dim=0
    )
    # recon_m_ps = ps.register_surface_mesh("recon", v_from_m(recon_m), f_from_m(gt_m), smooth_shade=True)
    # recon_m_ps.add_scalar_quantity("cd_error", cd_error.numpy(), defined_on='faces', enabled=True, cmap='jet')
    # ps.show()
    shading_ = {"flat": False, "colormap": "jet", "wireframe": True}
    p_ = mp.subplot(
        v_from_m(recon_m),
        f_from_m(gt_m),
        c=cd_error.numpy(),
        s=[1, 2, 0],
        shading=shading_,
    )
    mp.subplot(
        v_from_m(source_m), f_from_m(source_m), s=[1, 2, 1], data=p_, shading=shading_
    )


def grad_op_from_face_grad(face_fields, faces, n_vertices, order_style="F"):
    # Code borrowed from pyFM
    n_faces = faces.shape[0]
    row_indices = np.repeat(np.arange(n_faces), 3)
    col_indices = faces.flatten()

    if order_style == "F":
        In = np.concatenate(
            [row_indices, row_indices + n_faces, row_indices + 2 * n_faces]
        )
        Jn = np.tile(col_indices, 3)
        Vn = face_fields.flatten(order="F")

    else:
        In = np.concatenate([3 * row_indices, 3 * row_indices + 1, 3 * row_indices + 2])
        Jn = np.tile(col_indices, 3)
        Vn = face_fields.flatten(order="F")

    Gmat = csr_matrix((Vn, (In, Jn)), shape=(3 * n_faces, n_vertices))

    return Gmat


def read_vts(vts_dir, prefix):
    with open(osp.join(vts_dir, "%s.vts" % prefix), "r") as sm:
        src_corresp = [int(i.rstrip()) - 1 for i in sm.readlines()]
    return np.array(src_corresp)


def read_map(vts_dir, prefix):
    with open(osp.join(vts_dir, "%s.map" % prefix), "r") as sm:
        src_corresp = [int(i.rstrip()) - 1 for i in sm.readlines()]
    return np.array(src_corresp)


def invert_p2p(p2p):
    """
    # ALERT: If there are many to one, this will be degenerate.
    """
    s = np.ones(p2p.size, p2p.dtype)
    s[p2p] = np.arange(p2p.size)
    return s


def invert_and_compose_p2p(p2p_1, p2p_2):
    """
    Compose two P2P maps. Assume that each are permutation to a canonical model.
    # ALERT: If there are many to one, this will be degenerate.
    Args:
        p2p_1 (np.ndarray):
        p2p_2 (np.ndarray):
    """
    p2p_S_to_T = invert_p2p(p2p_1)[p2p_2]
    return p2p_S_to_T


def get_p2p_tup_from_vts(vts_dir, src, tar):
    src_canonical = read_vts(vts_dir, src)
    tar_canonical = read_vts(vts_dir, tar)
    p2p_tup = tuple(list(zip(src_canonical, tar_canonical)))
    return p2p_tup


def get_p2p_tup_from_map(vts_dir, src, tar):
    src_canonical = read_map(vts_dir, src)
    tar_canonical = read_map(vts_dir, tar)
    p2p_tup = tuple(list(zip(src_canonical, tar_canonical)))
    return p2p_tup


def get_ev(verts, faces, num_ev, return_mass=False):
    import robust_laplacian
    import scipy.sparse.linalg as sla

    L, M = robust_laplacian.mesh_laplacian(verts, faces, mollify_factor=1e-5)
    evals, evecs = sla.eigsh(L, 200, M, sigma=-0.01)
    if return_mass:
        return evals[:num_ev], evecs[:, :num_ev], M
    return evals[:num_ev], evecs[:, :num_ev]


def face_area_from_m(mesh):
    return trimesh.triangles.area(v_from_m(mesh)[f_from_m(mesh)])


def get_rotated_mesh(mesh, angle=90):
    """
    Rotate a mesh by angle degrees.
    """
    mesh_v = v_from_m(mesh)
    rot_mat = rotation_matrix(angle=np.radians(angle), direction=[1, 0, 0])[:3, :3]
    mesh_transf_ = trimesh_from_vf(mesh_v @ rot_mat, f_from_m(mesh))
    return mesh_transf_


def get_p2p_tup_from_vts_cross(base_dir, src, tar, cross_cat_map):
    src_vts_dir = osp.join(base_dir, Path(cross_cat_map).stem.split("_")[0], "corres")
    src_canonical = read_vts(src_vts_dir, src)
    tar_vts_dir = osp.join(base_dir, Path(cross_cat_map).stem.split("_")[1], "corres")
    tar_canonical = read_vts(tar_vts_dir, tar)
    cat_canonical = read_vts(Path(cross_cat_map).parent, Path(cross_cat_map).stem)
    p2p_tup = tuple(list(zip(src_canonical, tar_canonical[cat_canonical])))
    return p2p_tup


def check_centered_and_scaled(my_mesh):
    if not isinstance(my_mesh, trimesh.Trimesh):
        my_mesh = load_m(my_mesh)
    mean_extent = (my_mesh.vertices.max(axis=0) + my_mesh.vertices.min(axis=0)) / 2
    radius = np.max(np.sqrt(np.sum(my_mesh.vertices**2, axis=1)))
    assert np.allclose(mean_extent, 0)
    assert np.allclose(radius, 1)


def trimesh_from_vfc(v, f, c):
    m = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=c, process=False)
    return m


def visu_pc(pts, color="nipy_spectral_r"):
    # com = np.mean(pts, axis=0)
    com = np.array([0.0, 0.0, 1.0])
    radii = 0.5 * np.linalg.norm(pts - com, axis=1)
    # radii = np.sin(10.* np.arccos(np.clip(pts[:, 0], -1, 1)))
    cmap = trimesh.visual.interpolate(radii, color_map=color)
    return cmap


def plot_multi_mesh_corresp(
    meshes, cmaps=[], scale=False, disp=np.array([1.0, 0.0, 0.0])
):
    geometries = []
    m1 = meshes[0]
    cmap1 = visu_pc(m1.vertices)

    # Identity if not specified
    if len(cmaps) == 0:
        for j in range(1, len(meshes)):
            cmaps.append(np.arange(meshes[j].vertices.shape[0]))

    v_1 = scale_to_unit_sphere(v_from_m(m1)) if scale else v_from_m(m1)
    geometries.append(trimesh_from_vfc(v_1, f_from_m(m1), cmap1))

    for ind, p2p in enumerate(cmaps):
        if isinstance(p2p, np.ndarray) and p2p.ndim == 2:
            cmap2 = np.zeros((meshes[ind + 1].vertices.shape[0], 4))
            cmap2[p2p[:, 1]] = cmap1[p2p[:, 0]]
        else:
            cmap2 = cmap1[p2p]
        vert_i = (
            scale_to_unit_sphere(meshes[ind + 1].vertices)
            if scale
            else meshes[ind + 1].vertices
        )
        face_i = f_from_m(meshes[ind + 1])
        geometries.append(trimesh_from_vfc(vert_i + (disp * (ind + 1)), face_i, cmap2))
    return trimesh.Scene(geometry=geometries)


def Mat_F_to_C(matrix_array):
    # From O1 to O2
    # Given in F order, reshape to C order
    # XXX, YYY, ZZZ --> XYZ, XYZ, XYZ
    num_groups = matrix_array.shape[0] // 3
    reshaped = matrix_array.reshape(3, num_groups, 3)
    transposed = reshaped.transpose(1, 0, 2)
    return transposed.reshape(matrix_array.shape)


# From O2 to O1
def Mat_C_to_F(matrix_array):
    # O2 to O1
    # Given in C order, reshape to F order
    # XYZ, XYZ, XYZ --> XXX, YYY, ZZZ
    num_repeats = matrix_array.shape[0] // 3
    reshaped = matrix_array.reshape(num_repeats, -1, 3)
    transposed = reshaped.transpose(1, 0, 2)
    return transposed.reshape(matrix_array.shape)


def face_vector_to_Mat_F(face_vectors):
    """
    Given a face vector field, convert it into a sparse matrix.
    The matrix is in F order, i.e, [XXX, YYY, ZZZ]

    Output can easily be left multiplied with gradient matrix.
    """
    f_num = face_vectors.shape[0]

    # Row indices: [0, 1, 2, ..., F-1, 0, 1, 2, ..., F-1, 0, 1, 2, ..., F-1]
    row_indices = np.tile(np.arange(f_num), 3)

    # Column indices: [0, 1, 2, ..., F-1, F, F+1, ..., 2F-1, 2F, 2F+1, ..., 3F-1]
    col_indices = np.concatenate(
        [np.arange(f_num), np.arange(f_num, 2 * f_num), np.arange(2 * f_num, 3 * f_num)]
    )
    # Data values are just the flattened version of F
    data_values = face_vectors.flatten()
    face_vf_mat = csr_matrix(
        (data_values, (row_indices, col_indices)), shape=(f_num, 3 * f_num)
    )
    return face_vf_mat


def get_default_directories(dataset_name):
    if dataset_name.lower() == "faust_o":
        return "/mnt/disk2/ramana/data/MPI-FAUST/training/registrations/"
    elif dataset_name.lower() == "faust_r":
        return (
            "/mnt/disk2/ramana/data/SGA18_orientation_BCICP_dataset/Dataset/FAUST/ply/"
        )
    elif dataset_name.lower() == "dt4d":
        return "/mnt/disk2/ramana/data/3DV22_DeformingThings4DMatching_dataset/DeformingThings4DMatching/"
    else:
        raise ValueError("Unknown dataset name")


def get_default_shape_pairs(dataset_name, full_test=False, isometric=False, **kwargs):
    if dataset_name.lower() == "faust_o" or dataset_name.lower() == "faust_r":
        shape_base_name = "tr_reg_{:03d}"
        if full_test:
            combined_range = range(80, 100)
            combo = list(combinations(combined_range, 2))
            return [
                (shape_base_name.format(i), shape_base_name.format(j)) for i, j in combo
            ]
        else:
            if isometric:
                combo = list(combinations(range(80, 90), 2))
                return [
                    (shape_base_name.format(i), shape_base_name.format(j))
                    for i, j in combo
                ]
            else:
                combo = product(range(80, 85), range(95, 100))
                return [
                    (shape_base_name.format(i), shape_base_name.format(j))
                    for i, j in combo
                ]

    elif dataset_name.lower() == "dt4d":
        assert kwargs["category_1"] is not None
        assert kwargs["data_dir"] is not None
        assert kwargs["category_2"] is not None
        base_dir = kwargs["data_dir"]
        cat_1_name = kwargs["category_1"]
        # can also be same as cat_1_name
        cat_2_name = kwargs["category_2"]
        src_meshes = sorted(glob(osp.join(base_dir, "%s" % cat_1_name, "*.obj")))
        tar_meshes = sorted(glob(osp.join(base_dir, "%s" % cat_2_name, "*.obj")))
        if not full_test:
            src_meshes = src_meshes[:10]
            tar_meshes = tar_meshes[-10:]

        combo = product(src_meshes, tar_meshes)
        return [(Path(i).stem, Path(j).stem) for i, j in combo]

    else:
        raise ValueError("Unknown dataset name")


def point_to_line_segment(P, A, B):
    AB = B - A
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0, 1)
    closest_point = A + t * AB
    return closest_point


def project_point_on_triangle(P, A, B, C):
    # Calculate the triangle's normal
    normal = np.cross(B - A, C - A)

    # Normalize the normal
    normal = normal / np.linalg.norm(normal)

    # Distance from P to the plane of the triangle
    distance = np.dot(P - A, normal)

    # Project the point onto the plane
    P_proj = P - distance * normal

    # Check if P_proj lies inside triangle using barycentric coordinates
    v0 = B - A
    v1 = C - A
    v2 = P_proj - A
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    alpha = (d11 * d20 - d01 * d21) / denom
    beta = (d00 * d21 - d01 * d20) / denom
    gamma = 1.0 - alpha - beta

    # If point lies inside the triangle, return it
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return P_proj

    # Else find the closest point on each of the triangle's edges
    closest_on_AB = point_to_line_segment(P_proj, A, B)
    closest_on_BC = point_to_line_segment(P_proj, B, C)
    closest_on_CA = point_to_line_segment(P_proj, C, A)

    # Find which is closest to P_proj
    distances = [
        np.linalg.norm(P_proj - closest_on_AB),
        np.linalg.norm(P_proj - closest_on_BC),
        np.linalg.norm(P_proj - closest_on_CA),
    ]
    closest_points = [closest_on_AB, closest_on_BC, closest_on_CA]

    return closest_points[np.argmin(distances)]


def eye_like(tensor):
    assert tensor.shape[-1] == tensor.shape[-2]
    if tensor.ndim == 4:
        b = tensor.shape[0]
        n = tensor.shape[1]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = eyed_tensor.unsqueeze(0).unsqueeze(0).expand(b, n, -1, -1)
    elif tensor.ndim == 3:
        n = tensor.shape[0]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = eyed_tensor.unsqueeze(0).expand(n, -1, -1)
    elif tensor.ndim == 5:
        b = tensor.shape[0]
        n = tensor.shape[1]
        m = tensor.shape[2]
        eyed_tensor = torch.eye(tensor.shape[-1]).to(tensor.device)
        expanded_eye = (
            eyed_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(b, n, m, -1, -1)
        )

    return expanded_eye.to(tensor.dtype)


def get_p2p_from_deform(deform_v, tar_v):
    tree = KDTree(tar_v)
    _, p2p = tree.query(deform_v)
    return p2p.squeeze()


def plot_mesh_corresp(m1, m2, cmap=None, scale=True, disp=np.array([1.0, 0.0, 0.0])):
    if isinstance(m1, str):
        m1 = trimesh.load(m1, process=False)
    if isinstance(m2, str):
        m2 = trimesh.load(m2, process=False)
    cmap1 = visu_pc(m1.vertices)
    if cmap is None:
        cmap2 = cmap1
    elif isinstance(cmap, np.ndarray) or isinstance(cmap, list):
        cmap2 = cmap1[cmap]
    else:
        assert isinstance(cmap, tuple)
        cmap_ar = np.array(cmap)
        cmap2 = np.zeros((v_from_m(m2).shape[0], 4))
        cmap2[cmap_ar[:, 1]] = cmap1[cmap_ar[:, 0]]
    m1_vert = v_from_m(m1) if not scale else scale_to_unit_sphere(v_from_m(m1))
    m2_vert = v_from_m(m2) if not scale else scale_to_unit_sphere(v_from_m(m2))
    return trimesh.Scene(
        geometry=[
            trimesh.Trimesh(
                vertices=scale_to_unit_sphere(m1_vert),
                faces=m1.faces,
                vertex_colors=cmap1,
                process=False,
            ),
            trimesh.Trimesh(
                vertices=m2_vert + disp,
                faces=m2.faces,
                vertex_colors=cmap2,
                process=False,
            ),
        ]
    )


def get_closest_f_dist(query_pt, gen_trigs):
    """
    Get distance to closest face using pytorch3d
    """
    from pytorch3d.loss.point_mesh_distance import _PointFaceDistance

    pts_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
    trig_idx = torch.tensor([0], dtype=torch.int64, device="cuda")
    max_points = query_pt.shape[0]
    min_area = 0.0
    point_face_distance = _PointFaceDistance.apply
    nn_dist = point_face_distance(
        query_pt, pts_idx, gen_trigs, trig_idx, max_points, min_area
    )
    return (nn_dist + 1e-10).sqrt()


def print_jacobian_stats(J_C, face_basis):
    J_C_proj = face_basis @ J_C
    _, sv_proj, _ = np.linalg.svd(J_C_proj)
    big_shrink_idx = np.nonzero(sv_proj < 1e-3)[0]
    big_expan_idx = np.nonzero(sv_proj > 5.0)[0]
    inv_idx = np.nonzero(sv_proj < 0)[0]
    print(
        "Number of shrink, expansion, inversions are {}, {}, {}".format(
            big_shrink_idx.shape, big_expan_idx.shape, inv_idx.shape
        )
    )


def sp_to_torch(sp_matrix):
    if not isinstance(sp_matrix, coo_matrix):
        sp_matrix = sp_matrix.tocoo()
    values = sp_matrix.data
    indices = np.vstack((sp_matrix.row, sp_matrix.col))
    sp_i = torch.LongTensor(indices)
    sp_v = torch.FloatTensor(values)
    sp_shape = sp_matrix.shape
    return torch.sparse.FloatTensor(sp_i, sp_v, torch.Size(sp_shape))


def safe_make_dirs(cur_dir):
    if not osp.isdir(cur_dir):
        os.makedirs(cur_dir)


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


def project_point_on_closest_trig(query, trigs, trig_N):
    assert query.shape[0] == 3, "Query should be 3D"
    query_expan = query[np.newaxis, :]
    A = trigs[:, 0, :]
    B = trigs[:, 1, :]
    C = trigs[:, 2, :]
    # Distance from P to the plane of all triangles
    distance = np.einsum("ij,ij->i", query_expan - A, trig_N)
    # get the closest triangle
    closest_tri_idx = np.argmin(np.abs(distance))
    distance = distance[closest_tri_idx]
    trig_N = trig_N[closest_tri_idx]
    A, B, C = trigs[closest_tri_idx]
    # Project the point onto the plane
    P_proj = query - distance * trig_N
    # Check if P_proj lies inside triangle using barycentric coordinates
    u_ = B - A
    v_ = C - A
    w_ = P_proj - A
    n_vec = np.cross(u_, v_)
    n_vec = n_vec / np.linalg.norm(n_vec)
    gamma = np.dot(np.cross(u_, w_), n_vec) / np.dot(n_vec, n_vec)
    beta = np.dot(np.cross(w_, v_), n_vec) / np.dot(n_vec, n_vec)
    alpha = 1 - gamma - beta

    # If point lies inside the triangle, return it
    if 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1:
        return P_proj

    # Else find the closest point on each of the triangle's edges
    closest_on_AB = point_to_line_segment(P_proj, A, B)
    closest_on_BC = point_to_line_segment(P_proj, B, C)
    closest_on_CA = point_to_line_segment(P_proj, C, A)
    # Find which is closest to P_proj
    distances = [
        np.linalg.norm(P_proj - closest_on_AB),
        np.linalg.norm(P_proj - closest_on_BC),
        np.linalg.norm(P_proj - closest_on_CA),
    ]
    closest_points = [closest_on_AB, closest_on_BC, closest_on_CA]

    return closest_points[np.argmin(distances)]


def construct_R(theta, kx, ky, kz):
    R = np.array(
        [
            [
                np.cos(theta) + kx**2 * (1 - np.cos(theta)),
                kx * ky * (1 - np.cos(theta)) - kz * np.sin(theta),
                kx * kz * (1 - np.cos(theta)) + ky * np.sin(theta),
            ],
            [
                ky * kx * (1 - np.cos(theta)) + kz * np.sin(theta),
                np.cos(theta) + ky**2 * (1 - np.cos(theta)),
                ky * kz * (1 - np.cos(theta)) - kx * np.sin(theta),
            ],
            [
                kz * kx * (1 - np.cos(theta)) - ky * np.sin(theta),
                kz * ky * (1 - np.cos(theta)) + kx * np.sin(theta),
                np.cos(theta) + kz**2 * (1 - np.cos(theta)),
            ],
        ]
    )
    return R


def construct_R_th(theta, kx, ky, kz):
    R = torch.zeros((3, 3))
    R[0, 0] = torch.cos(theta) + kx**2 * (1 - torch.cos(theta))
    R[0, 1] = kx * ky * (1 - torch.cos(theta)) - kz * torch.sin(theta)
    R[0, 2] = kx * kz * (1 - torch.cos(theta)) + ky * torch.sin(theta)
    R[1, 0] = ky * kx * (1 - torch.cos(theta)) + kz * torch.sin(theta)
    R[1, 1] = torch.cos(theta) + ky**2 * (1 - torch.cos(theta))
    R[1, 2] = ky * kz * (1 - torch.cos(theta)) - kx * torch.sin(theta)
    R[2, 0] = kz * kx * (1 - torch.cos(theta)) - ky * torch.sin(theta)
    R[2, 1] = kz * ky * (1 - torch.cos(theta)) + kx * torch.sin(theta)
    R[2, 2] = torch.cos(theta) + kz**2 * (1 - torch.cos(theta))
    return R


def rotation_about_normal(normal, theta, with_scale=False, is_torch=False):
    # Normalize the rotation axis
    if not is_torch:
        kx, ky, kz = normal[:, 0], normal[:, 1], normal[:, 2]
        R_3x3 = np.zeros((normal.shape[0], 3, 3))
        for i in range(normal.shape[0]):
            R_3x3[i] = construct_R(theta[i], kx[i], ky[i], kz[i])

    if is_torch:
        R_3x3 = torch.zeros((normal.shape[0], 3, 3)).to(theta.device)
        normals_th = torch.from_numpy(normal)
        kx, ky, kz = normals_th[:, 0], normals_th[:, 1], normals_th[:, 2]
        S_3x3 = torch.eye(3).repeat(normal.shape[0], 1, 1).to(theta.device)
        for i in range(normal.shape[0]):
            if with_scale:
                R_3x3[i] = construct_R_th(
                    theta[i][0], normal[i, 0], normal[i, 1], normal[i, 2]
                )
                S_3x3[i] = torch.diag(torch.tensor([theta[i][1], theta[i][2], 1.0]))
            else:
                R_3x3[i] = construct_R_th(
                    theta[i], normal[i, 0], normal[i, 1], normal[i, 2]
                )
        R_3x3 = S_3x3 @ R_3x3
    # Compute the 3x3 rotation matrix using Rodrigues' formula
    return R_3x3


def scaled_rotation_matrix(
    N_F, F_N, max_angle_deg=90, min_angle_deg=-90.0, scale_type="uniform"
):
    """
    Returns F random 3x2 rotation matrices scaled by a random scale factor.
    The rotation is strictly between -90 and 90 degrees.
    """
    # angles_rad = np.random.uniform(-np.radians(max_angle_deg), np.radians(max_angle_deg), size=F)
    angles_rad = np.random.uniform(
        np.radians(min_angle_deg), np.radians(max_angle_deg), size=N_F
    )
    if scale_type == "uniform":
        scales = (
            np.random.uniform(size=N_F) + 1
        )  # Random scales > 1 for the sake of this example
    elif scale_type == "constant":
        scales = 1.0
    cosines = scales * np.cos(angles_rad)
    sines = scales * np.sin(angles_rad)

    # Initialize the rotation matrices
    rotation_matrices = np.zeros((N_F, 3, 2))
    rotation_matrices[:, 0, 0] = cosines
    rotation_matrices[:, 1, 0] = sines
    rotation_matrices[:, 0, 1] = -sines
    rotation_matrices[:, 1, 1] = cosines

    rotation_matrices_3x3 = rotation_about_normal(F_N, angles_rad)

    return rotation_matrices, rotation_matrices_3x3


def zero_center_pts(pts):
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2
    return pts


def to_3D(pts):
    return np.c_[pts, np.zeros(pts.shape[0])]


def fix_and_solve_(src_v, src_f, Jac_mat, pin_v, v_idx=1000):
    my_m = MyMesh(verts=src_v, faces=src_f, scale_type=None)
    my_m.process_diff_ops()
    Lap_mat = my_m.lap_op_0.copy()
    Lap_mat[v_idx, :] = 0.0
    Lap_mat[v_idx, v_idx] = 1.0
    rhs_mat = my_m.div_C @ (Jac_mat.reshape(-1, 3))
    rhs_mat[v_idx, :] = pin_v
    rhs_mat = coo_matrix(rhs_mat).tocsc()
    Lap_mat = coo_matrix(Lap_mat).tocsc()
    return spsolve(Lap_mat, rhs_mat).toarray()


def retrieve_angle_2x2(rotation_matrix):
    theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.degrees(theta)  # Returns the angle in degrees


def get_adjacent_face_idx(faces, f_idx):
    face_edge_1 = faces[:, [1, 2]]
    face_edge_2 = faces[:, [2, 0]]
    face_edge_3 = faces[:, [0, 1]]
    face_edges = np.stack((face_edge_1, face_edge_2, face_edge_3), axis=1)
    half_edge_tree = KDTree(igl.unique_edge_map(faces)[0])
    incident_f_idx = []
    for f_idx_ in f_idx:
        f_edges = face_edges[f_idx_].reshape(-1, 2)
        for v1_idx, v2_idx in f_edges:
            d_, f_ = half_edge_tree.query((v1_idx, v2_idx))
            if d_ == 0 and f_ % faces.shape[0] != f_idx_:
                incident_f_idx.append(f_ % faces.shape[0])
            d_, f_ = half_edge_tree.query((v2_idx, v1_idx))
            if d_ == 0 and f_ % faces.shape[0] != f_idx_:
                incident_f_idx.append(f_ % faces.shape[0])
    return incident_f_idx


def total_rotation_around_v_from_J(v, f, J_2x2):
    grad_mat = igl.grad(v, f)
    vert_angles = defaultdict(float)
    for v_idx in range(v.shape[0]):
        nz_idx = grad_mat[: f.shape[0], v_idx].nonzero()[0]
        for nz_idx_ in nz_idx:
            vert_angles[v_idx] += retrieve_angle_2x2(J_2x2[nz_idx_])
    vert_angles_ar = np.array([vert_angles[v_idx] for v_idx in range(v.shape[0])])
    return vert_angles_ar


def get_2x2_rotmat_from_angles(angles):
    R_2x2 = np.zeros((angles.shape[0], 2, 2))
    R_2x2[:, 0, 0] = np.cos(angles)
    R_2x2[:, 1, 0] = np.sin(angles)
    R_2x2[:, 0, 1] = -np.sin(angles)
    R_2x2[:, 1, 1] = np.cos(angles)
    return R_2x2


def get_2x2_rotmat_from_angles_th(angles):
    R_2x2 = torch.zeros((angles.shape[0], 2, 2), device=angles.device)
    R_2x2[:, 0, 0] = torch.cos(angles)
    R_2x2[:, 1, 0] = torch.sin(angles)
    R_2x2[:, 0, 1] = -torch.sin(angles)
    R_2x2[:, 1, 1] = torch.cos(angles)
    return R_2x2


def Mat_F_to_C_np(matrix_np, n_f=2):
    # From O1 to O2
    # Given in F order, reshape to C order
    # XXX, YYY, ZZZ --> XYZ, XYZ, XYZ
    num_groups = matrix_np.shape[0] // 3
    reshaped = matrix_np.reshape(3, num_groups, n_f)
    transposed = np.transpose(reshaped, (1, 0, 2))
    return transposed.reshape(matrix_np.shape)


def Mat_F_to_C_sp(mat_sp):
    # Given in F order, reshape to C order
    # XXX, YYY, ZZZ --> XYZ, XYZ, XYZ
    mat_sp_x = mat_sp[: mat_sp.shape[0] // 3, :]
    mat_sp_y = mat_sp[mat_sp.shape[0] // 3 : 2 * mat_sp.shape[0] // 3, :]
    mat_sp_z = mat_sp[2 * mat_sp.shape[0] // 3 :, :]
    mat_sp_c = sp.hstack([mat_sp_x, mat_sp_y, mat_sp_z]).reshape(-1, mat_sp.shape[1])
    return mat_sp_c


def Mat_C_to_F_np(matrix_np, n_f=2):
    # Given in C order, reshape to F order
    # XYZ, XYZ, XYZ --> XXX, YYY, ZZZ
    num_repeats = matrix_np.shape[0] // 3
    reshaped = matrix_np.reshape(num_repeats, -1, n_f)
    transposed = np.transpose(reshaped, (1, 0, 2))
    return transposed.reshape(matrix_np.shape)


def normalize_np(x: np.array, eps: float = 1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def get_rot_scale_shear_about_normal(normal, b_1, b_2, theta, s_x, s_y, sh_x, sh_y):
    # Normalize the rotation axis
    kx, ky, kz = normal[:, 0], normal[:, 1], normal[:, 2]
    R_3x3 = np.zeros((normal.shape[0], 3, 3))
    for i in range(normal.shape[0]):
        R_3x3[i] = construct_R(theta[i], kx[i], ky[i], kz[i])
    S_2x2 = np.tile(np.eye(2), (normal.shape[0], 1, 1))
    Sh_2x2 = np.tile(np.eye(2), (normal.shape[0], 1, 1))
    Sh_2x2[:, 0, 1] = sh_x
    Sh_2x2[:, 1, 0] = sh_y
    S_2x2[:, 0, 0] = s_x
    S_2x2[:, 1, 1] = s_y
    M_2x2 = Sh_2x2 @ S_2x2
    transformation_3d = np.tile(np.eye(3), (normal.shape[0], 1, 1))
    for k in range(normal.shape[0]):
        for i in range(3):
            for j in range(3):
                u = b_1[k]
                v = b_2[k]
                transformation_3d[k, i, j] += (
                    u[i] * u[j] * (M_2x2[k, 0, 0] - 1)
                    + v[i] * v[j] * (M_2x2[k, 1, 1] - 1)
                    + (u[i] * v[j] + v[i] * u[j]) * M_2x2[k, 0, 1]
                )

    M = transformation_3d @ R_3x3
    return M


def get_incenter(trig):
    # (aA + bB + cC)/(a+b+c)
    # note, a = Edgelen opp to A
    trig_abc_l = [
        th_norm(trig[:, i2, :] - trig[:, i1, :], dim=1)
        for i1, i2 in zip([1, 2, 0], [2, 0, 1])
    ]
    trig_abc_th = torch.stack(trig_abc_l).T + 1e-8
    trig_abc_sum = trig_abc_th.sum(dim=1, keepdim=True)
    A = trig[:, 0, :]
    B = trig[:, 1, :]
    C = trig[:, 2, :]
    incenter = (
        trig_abc_th[:, :1] * A + trig_abc_th[:, 1:2] * B + trig_abc_th[:, 2:3] * C
    )
    return incenter / (trig_abc_sum + 1e-12)


def custom_sample_trig(gen_trig):
    """
    Points are sampled along perpendicular bisector and
    the line connecting median to vertices.
    """

    def rand_unif_th(shape, start, end, device="cuda"):
        return torch.FloatTensor(shape).uniform_(start, end).to(device).float()

    in_center = get_incenter(gen_trig)
    median = gen_trig.mean(dim=1)
    a_coord = gen_trig[:, 0, :]
    b_coord = gen_trig[:, 1, :]
    c_coord = gen_trig[:, 2, :]
    ab_coord = 0.5 * (a_coord + b_coord)
    bc_coord = 0.5 * (b_coord + c_coord)
    ca_coord = 0.5 * (c_coord + a_coord)

    n_pts = gen_trig.shape[0]
    r_1 = rand_unif_th((n_pts), 0.2, 0.8).unsqueeze(-1)
    r_2 = rand_unif_th((n_pts), 0.2, 0.8).unsqueeze(-1)
    r_3 = rand_unif_th((n_pts), 0.2, 0.8).unsqueeze(-1)

    mp_1 = r_1 * a_coord + (1 - r_1) * median
    mp_2 = r_2 * b_coord + (1 - r_2) * median
    mp_3 = r_3 * c_coord + (1 - r_3) * median
    icp_1 = r_3 * ab_coord + (1 - r_3) * in_center
    icp_2 = r_1 * bc_coord + (1 - r_1) * in_center
    icp_3 = r_2 * ca_coord + (1 - r_2) * in_center

    samp_pts = [
        mp_1,
        mp_2,
        mp_3,
        icp_1,
        icp_2,
        icp_3,
        median,
        a_coord,
        b_coord,
        c_coord,
    ]
    samp_f_idx = (
        torch.arange(gen_trig.shape[0]).repeat_interleave(len(samp_pts)).view(-1, 1)
    )

    samp_pts_stack = torch.stack(samp_pts, dim=1).view(-1, 3)
    return samp_pts_stack, samp_f_idx


def plot_vars(all_lists, labels):
    plt.figure()
    for pl, label in zip(all_lists, labels):
        plt.plot(pl)
    plt.legend(labels)
    plt.show()


def get_proj_coeff(Phi, Mass, func):
    return np.linalg.inv(Phi.T @ Mass @ Phi) @ Phi.T @ Mass @ func


def get_proj_coeff_complex(Phi_c, Mass, vf_c):
    return np.linalg.inv(Phi_c.conj().T @ Mass @ Phi_c) @ Phi_c.conj().T @ Mass @ vf_c


def get_mat_from_complex(A):
    real_part = np.real(A)
    imag_part = np.imag(A)
    # Stack them along the third dimension
    A_split = np.stack((real_part, imag_part), axis=-1)
    return A_split


def get_only_rotation_th(cov_mat):
    U, _, Vh = torch.linalg.svd(cov_mat)
    # Compute rotation
    rot = U @ Vh
    # Fix reflection
    det_rot = torch.linalg.det(rot)
    U_fix = U.clone()
    U_fix[det_rot < 0, :, -1] *= -1
    rot = U_fix @ Vh
    return rot


def get_fps(pts, K, mask=None):
    """
    Farthest point sampling alongside indices
    """

    def l2_distance(p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    farthest_pts = np.zeros((K, 3))
    if mask is not None:
        pts_ma = pts[mask]
    else:
        pts_ma = pts
    first_ind = np.random.randint(len(pts_ma))
    farthest_pts[0] = pts_ma[first_ind]
    distances = l2_distance(farthest_pts[0], pts_ma)
    pt_indices = [first_ind]
    for i in range(1, K):
        arg_max_ind = np.argmax(distances)
        farthest_pts[i] = pts_ma[arg_max_ind]
        pt_indices.append(arg_max_ind)
        distances = np.minimum(distances, l2_distance(farthest_pts[i], pts_ma))

    # Put a tree if there is a mask
    if mask is not None:
        tree = KDTree(pts)
        _, pt_indices = tree.query(farthest_pts)

    return farthest_pts, pt_indices


def spth_to_cholespy(sp_th):
    inds_th = sp_th.coalesce().indices().cpu()
    vals_th = sp_th.coalesce().values().cpu().double()
    n = sp_th.shape[0]
    return CholeskySolverD(n, inds_th[0, :], inds_th[1, :], vals_th, MatrixType.COO)


def get_curvature_weighted_fps(pts, g, K, alpha=2.0):
    """
    Farthest point sampling alongside indices, weighted by Gaussian curvature.
    """

    def l2_distance(p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    # Normalize the Gaussian curvature values to [0, 1]
    g_normalized = (g - g.min()) / (g.max() - g.min()) * alpha

    farthest_pts = np.zeros((K, 3))
    first_ind = np.random.randint(len(pts))
    farthest_pts[0] = pts[first_ind]
    distances = l2_distance(farthest_pts[0], pts) * (1 + g_normalized)
    pt_indices = [first_ind]

    for i in range(1, K):
        arg_max_ind = np.argmax(distances)
        farthest_pts[i] = pts[arg_max_ind]
        pt_indices.append(arg_max_ind)
        distances = np.minimum(
            distances, l2_distance(farthest_pts[i], pts) * (1 + g_normalized)
        )

    return farthest_pts, pt_indices


def visu_scalar(v, f, scalar, color="nipy_spectral"):
    from Alfred.helper import trimesh_from_vfc

    cmap = trimesh.visual.interpolate(scalar, color_map=color)
    return trimesh_from_vfc(v, f, cmap)


def get_only_rotation_np(cov_mat):
    U, sigma, VT = np.linalg.svd(cov_mat)
    # Compute rotation
    rot = U @ VT
    U[np.linalg.det(rot) < 0, :, -1] *= -1
    rot = U @ VT
    return rot


def construct_fourth_vert_th(verts, faces):
    trig = verts[faces]
    centroid = torch.mean(trig, dim=1)
    trig_a, trig_b, trig_c = trig[:, 0, :], trig[:, 1, :], trig[:, 2, :]
    comp_1 = trig_b - trig_a
    comp_2 = trig_c - trig_a
    cross_pdt = torch.cross(comp_1, comp_2)
    fourth_vert = centroid + cross_pdt / torch.linalg.norm(
        cross_pdt, dim=1, keepdim=True
    )
    return fourth_vert


def get_extrinsic_frame_lifted(src_v, src_f, tet_v):
    trig_a, trig_b, trig_c = src_v[src_f[:, 0]], src_v[src_f[:, 1]], src_v[src_f[:, 2]]
    comp_1 = trig_b - trig_a
    comp_2 = trig_c - trig_a
    comp_3 = tet_v - trig_a
    return np.column_stack((comp_1, comp_2, comp_3)).reshape(-1, 3, 3)


def get_extrinsic_frame_tet(tet_v, tet_f):
    trig_d = tet_v[tet_f[:, 3]]
    trig_a, trig_b, trig_c = tet_v[tet_f[:, 0]], tet_v[tet_f[:, 1]], tet_v[tet_f[:, 2]]
    comp_1 = trig_b - trig_a
    comp_2 = trig_c - trig_a
    comp_3 = trig_d - trig_a
    return np.column_stack((comp_1, comp_2, comp_3)).reshape(-1, 3, 3)


def lift_jacobian_th(src_v, src_f, pred_Jac, v_e_inc, v_e_inc_fac, tet_faces):
    src_v, src_f = src_v.squeeze(), src_f.squeeze()
    src_fv = numpied(construct_fourth_vert_th(src_v, src_f))
    src_frame = get_extrinsic_frame_lifted(numpied(src_v), numpied(src_f), src_fv)
    tar_frame = src_frame @ numpied(pred_Jac)
    def_v = v_e_inc_fac(v_e_inc.T @ tar_frame.reshape(-1)).reshape(-1, 3)
    def_v = zero_center_pts(def_v)
    # tet verts are [v, f_v, v_v]
    v_v = numpied(src_v) + igl.per_vertex_normals(numpied(src_v), numpied(src_f)) * 0.2
    src_tet_v = np.r_[numpied(src_v), numpied(src_fv), v_v]
    tet_v_v = (
        def_v[: src_v.shape[0]]
        + igl.per_vertex_normals(def_v[: src_v.shape[0]], numpied(src_f)) * 0.2
    )
    tar_tet_v = np.r_[def_v, tet_v_v]

    src_tet_frame = get_extrinsic_frame_tet(src_tet_v, tet_faces)
    tar_tet_frame = get_extrinsic_frame_tet(tar_tet_v, tet_faces)
    # Remove known jac
    tet_Jac = (
        np.linalg.inv(src_tet_frame[src_f.shape[0] :]) @ tar_tet_frame[src_f.shape[0] :]
    )
    tet_jac_th = torch.from_numpy(tet_Jac).float().to(pred_Jac.device)
    combined_jac = torch.cat((pred_Jac, tet_jac_th), dim=0)
    # Project to rotation and return.
    return get_only_rotation_th(combined_jac)[: src_f.shape[0]]


def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)
    ).coalesce()


# check for one shape then move to more.
def get_eye_th(dim_1, dim_2, device="cuda"):
    return torch.eye(dim_2).unsqueeze(0).repeat(dim_1, 1, 1).to(device)


def get_eye_like_np(mat):
    # mat: identity matrix of shape (n, m, m)
    return np.eye(mat.shape[1])[np.newaxis, :, :].repeat(mat.shape[0], axis=0)


def get_eye_like_th(mat, device="cuda"):
    # mat: identity matrix of shape (n, m, m)
    return torch.eye(mat.shape[1]).unsqueeze(0).repeat(mat.shape[0], 1, 1).to(device)


def batched_trace_th(A):
    return torch.einsum("bii->b", A)


def polar_decomp_np(my_mat, mode="right"):
    # Polar decomposition in terms of singular-value decomposition
    if my_mat.ndim == 2:
        U, S, Vh = np.linalg.svd(my_mat)
        u = U @ Vh
        p = Vh.T.conj() @ np.diag(S) @ Vh
    elif my_mat.ndim == 3:
        U, S, Vh = np.linalg.svd(my_mat)
        u = U @ Vh
        diag_S = get_eye_like_np(S) * S[:, np.newaxis, :]
        if mode == "right":
            p = Vh.transpose(0, 2, 1).conj() @ diag_S @ Vh
        else:
            p = U @ diag_S @ U.transpose(0, 2, 1).conj()
    else:
        raise ValueError("Input tensor must be 2 or 3 dimensional")
    return u, p


def polar_decomp_th(my_mat):
    # Polar decomposition in terms of singular-value decomposition
    if my_mat.ndim == 2:
        U, S, Vh = torch.linalg.svd(my_mat)
        u = U @ Vh
        p = Vh.T.conj() @ torch.diag(S) @ Vh
    elif my_mat.ndim == 3:
        U, S, Vh = torch.linalg.svd(my_mat)
        u = U @ Vh
        diag_S = get_eye_like_th(S) * S.unsqueeze(1)
        p = Vh.transpose(2, 1).conj() @ diag_S @ Vh
    else:
        raise ValueError("Input tensor must be 2 or 3 dimensional")
    return u, p


def sinkhorn(d, sigma=0.1, num_sink=10):
    d = d / d.mean()

    log_p = -d / (2 * sigma**2)

    for it in range(num_sink):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True)
    p = torch.exp(log_p)
    log_p = log_p - torch.logsumexp(log_p, dim=0, keepdim=True)
    p_adj = torch.exp(log_p).transpose(0, 1)
    return p, p_adj


def dist_mat(x, y, inplace=True):
    d = torch.mm(x, y.transpose(0, 1))
    v_x = torch.sum(x**2, 1).unsqueeze(1)
    v_y = torch.sum(y**2, 1).unsqueeze(0)
    d *= -2
    if inplace:
        d += v_x
        d += v_y
    else:
        d = d + v_x
        d = d + v_y

    return d


def convert_to_batch(tensor1, nb, dtype=torch.float32, device="cuda"):
    if isinstance(tensor1, np.ndarray):
        tensor1 = torch.from_numpy(tensor1).to(dtype).to(device)

    if tensor1.ndim == 2:
        tensor1 = tensor1.unsqueeze(0)
    if tensor1.shape[0] == nb:
        return tensor1
    tensor1 = tensor1.expand(nb, -1, -1).contiguous()
    return tensor1


def b3nTobn3(points):
    assert points.ndim == 3
    if not points.size(2) == 3:
        points = points.transpose(2, 1)
    return points


def get_dihedral_angle(v, f, edge_ar, edge_face_inc_ar):
    face_vecs = v[f]
    face_normals = torch.cross(
        face_vecs[:, 1] - face_vecs[:, 0], face_vecs[:, 2] - face_vecs[:, 0]
    )
    face_normals = face_normals / torch.norm(face_normals, dim=1).unsqueeze(1)
    edge_normals = torch.zeros_like(face_normals)
    edge_normals[edge_face_inc_ar[:, 0]] = face_normals[edge_face_inc_ar[:, 0]]
    edge_normals[edge_face_inc_ar[:, 1]] = face_normals[edge_face_inc_ar[:, 1]]
    edge_normals[edge_face_inc_ar[:, 2]] = face_normals[edge_face_inc_ar[:, 2]]
    edge_normals = edge_normals / torch.norm(edge_normals, dim=1).unsqueeze(1)
    dihedral_angles = torch.einsum(
        "bi,bi->b", edge_normals[edge_ar[:, 0]], edge_normals[edge_ar[:, 1]]
    )
    dihedral_angles = torch.acos(dihedral_angles)
    return dihedral_angles


def get_e_f_inc_ar(src_m):
    e_f_inc_ar = []
    for e in src_m.edges:
        v1, v2 = e
        e_f_inc_ar.append(src_m.get_half_edge_face(v1, v2))
        e_f_inc_ar.append(src_m.get_half_edge_face(v2, v1))
    e_f_inc_ar = np.array(e_f_inc_ar).reshape(-1, 2)
    return e_f_inc_ar


def zero_center_pts_th(verts):
    center = (verts.max(dim=0).values + verts.min(dim=0).values) / 2
    centered_verts = verts - center
    return centered_verts


def scale_to_unit_sphere_th(points, center=None, buffer=1.0):
    midpoints = (torch.max(points, dim=0)[0] + torch.min(points, dim=0)[0]) / 2
    #     midpoints = np.mean(points, axis=0)
    points = points - midpoints
    scale = torch.max(torch.sqrt(torch.sum(points**2, dim=1))) * buffer
    points = points / scale
    return points


def safe_frob_norm(th_1):
    return torch.linalg.norm(th_1 + 1e-8, ord="fro", dim=(-2, -1))


def fourier_encode(inp, embedding_size=64, embedding_scale=12.0):
    """
    Positional encoding for 3D points

    Args:
        inp (torch.Tensor): Input tensor of shape (B, N, 3)
        embedding_size (int): Size of the embedding
        embedding_scale (float): Scaling factor for the embedding

    Returns:
        stacked_x (torch.Tensor): Encoded tensor of shape (B, N, embedding_size)
    """
    bvals = 2.0 ** np.linspace(0, embedding_scale, embedding_size // 3) - 1.0
    bvals = (
        torch.from_numpy(
            np.reshape(np.eye(3) * bvals[:, None, None], [len(bvals) * 3, 3])
        )
        .float()
        .cuda()
    )
    avals = torch.ones_like(bvals[:, 0]).cuda()
    inp_coord = b3nTobn3(inp)
    stacked_x = torch.cat(
        [
            avals * torch.sin((2.0 * np.pi * inp_coord) @ torch.transpose(bvals, 0, 1)),
            avals * torch.cos((2.0 * np.pi * inp_coord) @ torch.transpose(bvals, 0, 1)),
        ],
        axis=-1,
    ) / torch.norm(avals)
    return stacked_x


def transfer_to_gpu(my_dict):
    for k, v in my_dict.items():
        if isinstance(v, torch.Tensor):
            my_dict[k] = v.cuda()
    return my_dict


def th_sp_diags(diagonal_array):
    """
    Create a sparse diagonal matrix in PyTorch given a diagonal array.

    Args:
        diagonal_array: A 1D tensor containing the diagonal values.

    Returns:
        A sparse tensor representing the diagonal matrix.
    """
    # Number of diagonal elements
    n = diagonal_array.size(0)

    # Create indices for the diagonal
    rows = torch.arange(n).to(diagonal_array.device)
    cols = torch.arange(n).to(diagonal_array.device)
    return SparseTensor(row=rows, col=cols, value=diagonal_array, sparse_sizes=(n, n))


def sp_den_mul(spTensor, denTensor):
    """
    Args:
        spTensor: torch_sparse tensor.
        denTensor: A dense tensor.
    Returns:
        A dense tensor.
    """
    from torch_scatter import scatter_add

    row, col, value = spTensor.coo()
    m = spTensor.sparse_sizes()[0]
    out = denTensor.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=-2, dim_size=m)
    return out


def sp_thsp_mul(spTensor_1, thspTensor_2):
    """
    Args:
        spTensor_1: torch_sparse tensor.
        thspTensor_2: Pytorch sparse Tensor.
    Returns:
        A dense tensor.
    """
    from torch_sparse.matmul import matmul
    from torch_sparse.tensor import SparseTensor

    indexB, valueB = thspTensor_2.coalesce().indices(), thspTensor_2.coalesce().values()
    n, k = spTensor_1.sparse_sizes()
    B = SparseTensor(row=indexB[0], col=indexB[1], value=valueB, sparse_sizes=(n, k))
    return matmul(spTensor_1, B)


def thsp_to_sp(th_sp):
    th_sp_ind = th_sp.coalesce().indices()
    rows, cols = th_sp_ind[0], th_sp_ind[1]
    values = th_sp.coalesce().values()
    size = th_sp.size()
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=size).to_device(
        th_sp.device
    )


def sp_to_thsp(sp_tensor):
    row, col, value = sp_tensor.coo()
    sp_size = sp_tensor.sparse_sizes()
    indices = torch.stack([row, col], dim=0)
    return torch.sparse_coo_tensor(indices, value, sp_size).to(col.device)


def th_speye(n, device="cuda"):
    idx = np.c_[np.arange(n), np.arange(n)].T
    return torch.sparse_coo_tensor(torch.from_numpy(idx), torch.ones(n), (n, n)).to(
        device
    )


def get_spth_from_idxs(out_d, t_shape, t_key):
    # B, V, 3F
    # shape = (out_d['src_v'].shape[0], out_d['src_v'].shape[1], out_d['src_f'].shape[1]*3)
    c_idx = out_d[t_key + "_cidx"]
    r_idx = out_d[t_key + "_ridx"]
    val = out_d[t_key + "_val"]
    batch_idx = torch.arange(len(c_idx)).repeat_interleave(c_idx.shape[1]).unsqueeze(0)
    batch_idx = torch.cat(
        (batch_idx, r_idx.reshape(1, -1), c_idx.reshape(1, -1)), dim=0
    )
    my_t = torch.sparse.FloatTensor(batch_idx.long(), val.flatten(), t_shape)
    return my_t


def sparse_np_to_tsp(A):
    Acoo = A.tocoo()
    values = torch.from_numpy(Acoo.data)
    row = torch.from_numpy(Acoo.row).long()
    col = torch.from_numpy(Acoo.col).long()
    shape = Acoo.shape
    return SparseTensor(row=row, col=col, value=values, sparse_sizes=shape)


def seamless_loss(batch_dict, pred_jac):
    e_f_inc = batch_dict["e_f_inc_ar"].long()
    src_edges = batch_dict["edges"].long()
    src_v = batch_dict["src_v"]
    pred_jac = pred_jac
    batch_idx = torch.arange(src_v.shape[0]).unsqueeze(1)
    src_e = (
        src_v[batch_idx, src_edges[:, :, 1], :]
        - src_v[batch_idx, src_edges[:, :, 0], :]
    )
    transf_1 = torch.einsum(
        "bed,bedf->bef", src_e, pred_jac[batch_idx, e_f_inc[:, :, 0]]
    )
    transf_2 = torch.einsum(
        "bed,bedf->bef", src_e, pred_jac[batch_idx, e_f_inc[:, :, 1]]
    )
    seamless_loss = torch.linalg.norm(transf_1 - transf_2, axis=-1).mean()
    return seamless_loss


def get_extrinsic_frame_ba(v, batch_dict):
    faces_ = batch_dict["src_f"]
    n_b = faces_.shape[0]
    batch_idx = torch.arange(n_b).unsqueeze(1).unsqueeze(2)
    trig = v[batch_idx, faces_]
    trig_a, trig_b, trig_c = trig[:, :, 0, :], trig[:, :, 1, :], trig[:, :, 2, :]
    comp_1 = trig_b - trig_a
    comp_2 = trig_c - trig_a
    comp_3 = vec_normalize_th(torch.cross(comp_1, comp_2))
    return torch.stack((comp_1, comp_2, comp_3), dim=3).permute(0, 1, 3, 2)


def get_extrinsic_frame(v, f):
    trig = v[f]
    trig_a, trig_b, trig_c = trig[:, 0, :], trig[:, 1, :], trig[:, 2, :]
    comp_1 = trig_b - trig_a
    comp_2 = trig_c - trig_a
    comp_3 = vec_normalize_th(torch.cross(comp_1, comp_2))
    return torch.stack((comp_1, comp_2, comp_3), dim=2).permute(0, 2, 1)


def get_jac_from_image_frame(vert_image, batch_dict):
    frame_inv = batch_dict["frame_inv"].cuda().squeeze()
    cur_frame = get_extrinsic_frame(vert_image, batch_dict["src_f"].squeeze())
    return frame_inv @ cur_frame


def get_jac_at_vert(verts, faces, Jac):
    jac_face = Jac.reshape(-1, 3, 3)
    jac_face_1 = jac_face[:, :, 0]
    jac_face_2 = jac_face[:, :, 1]
    jac_face_3 = jac_face[:, :, 2]
    jac_vert_1 = igl.average_onto_vertices(verts, faces, jac_face_1)
    jac_vert_2 = igl.average_onto_vertices(verts, faces, jac_face_2)
    jac_vert_3 = igl.average_onto_vertices(verts, faces, jac_face_3)
    jac_vert = np.column_stack((jac_vert_1, jac_vert_2, jac_vert_3)).reshape(-1, 3, 3)
    return jac_vert


def get_W_at_vert(verts, faces, W):
    jac_face = W.reshape(-1, 2, 2)
    jac_face_1 = jac_face[:, :, 0]
    jac_face_2 = jac_face[:, :, 1]
    jac_vert_1 = igl.average_onto_vertices(verts, faces, jac_face_1)
    jac_vert_2 = igl.average_onto_vertices(verts, faces, jac_face_2)
    jac_vert = np.column_stack((jac_vert_1, jac_vert_2)).reshape(-1, 2, 2)
    return jac_vert


def get_Q_at_vert(verts, faces, Q):
    q1 = igl.average_onto_vertices(verts, faces, Q[:, :3])
    q2 = igl.average_onto_vertices(verts, faces, Q[:, 1:])
    return np.c_[q1, q2[:, -1:]].reshape(-1, 4)


def get_p2p_non_is(evecs_x, evecs_y, mass_x, mass_y, p2p):
    evecs_trans_y = evecs_y.T @ mass_y
    evecs_trans_x = evecs_x.T @ mass_x
    # compute Pyx from functional map
    Cxy = evecs_trans_y @ evecs_x[p2p]
    Pyx = evecs_y @ Cxy @ evecs_trans_x
    return Pyx


def generate_tex_coords(verts, col1=1, col2=0, mult_const=1):
    ind = np.argsort(np.std(verts, axis=0))[::-1]
    verts = verts[:, ind]
    vt = np.stack([verts[:, col1], verts[:, col2]], axis=-1)
    vt -= np.min(vt, axis=0, keepdims=True)
    vt = mult_const * vt / np.max(vt)
    vt[:, 0] = (vt[:, 0] - vt[:, 0].min()) / (vt[:, 0].max() - vt[:, 0].min())
    vt[:, 1] = (vt[:, 1] - vt[:, 1].min()) / (vt[:, 1].max() - vt[:, 1].min())
    return vt


def write_obj_pair(
    file_name1, file_name2, verts1, faces1, verts2, faces2, Pyx, texture_file
):
    # write off for shape 1
    uv1 = generate_tex_coords(verts1)
    if not os.path.exists(file_name1):
        write_obj_with_texture(verts1, faces1, file_name1, uv1, texture_file)

    # write off for shape 2
    if Pyx.ndim == 2:
        uv2 = Pyx @ uv1
    else:
        assert Pyx.ndim == 1
        uv2 = uv1[Pyx]
    write_obj_with_texture(verts2, faces2, file_name2, uv2, texture_file)


def write_texture_from_map(src_m, tar_m, p2p, save_dir, texture_file):
    if not hasattr(src_m, "vertex_mass"):
        src_m.process_diff_ops(fac_lap=True)
    if not hasattr(tar_m, "vertex_mass"):
        tar_m.process_diff_ops(fac_lap=True)
    if not hasattr(tar_m, "lbo_evecs"):
        tar_m.process_lbo_basis(n_ev=100)
    elif tar_m.lbo_evecs.shape[1] != 100:
        tar_m.process_lbo_basis(n_ev=100)
    if not hasattr(src_m, "lbo_evecs"):
        src_m.process_lbo_basis(n_ev=100)
    elif src_m.lbo_evecs.shape[1] != 100:
        src_m.process_lbo_basis(n_ev=100)
    Pyx = get_p2p_non_is(
        src_m.lbo_evecs, tar_m.lbo_evecs, src_m.vertex_mass, tar_m.vertex_mass, p2p
    )
    file_x = osp.join(save_dir, src_m.name_str + ".obj")
    file_y = osp.join(save_dir, src_m.name_str + "-" + tar_m.name_str + ".obj")
    write_obj_pair(
        file_x,
        file_y,
        src_m.verts,
        src_m.faces,
        tar_m.verts,
        tar_m.faces,
        Pyx,
        texture_file,
    )


def write_obj_with_texture(verts, faces, file_name, uv, texture_name):
    """
    write .obj file with texture.
    Args:
        verts (np.ndarray): vertices. [V, 3].
        faces (np.ndarray): faces. [F, 3].
        uv (np.ndarray, None): texture maps. [V, 2]
        texture_name (str): texture map image file name.
        file_name (str): stored file name.
    """
    assert (
        verts.shape[-1] == 3
    ), f"vertex does not have the correct format: {verts.shape}."
    assert (
        faces.shape[-1] == 3
    ), f"face does not have the correct format: {faces.shape}."
    if uv is not None:
        assert (
            uv.shape[-1] == 2
        ), f"vertex texture does not have the correct format: {uv.shape}."
    object_name = os.path.splitext(os.path.basename(file_name))[0]
    faces = faces.astype(int)
    with open(file_name, "w") as f:
        # head
        f.write("# write_obj (c) 2004 Gabriel Peyr\n")
        f.write(f"mtllib ./{object_name}.mtl\n")
        f.write(f"g\n# object {object_name} to come\n")

        # vertex position
        f.write(f"# {verts.shape[0]} vertex\n")
        for i in range(verts.shape[0]):
            f.write(f"v {verts[i][0]:.6f} {verts[i][1]:.6f} {verts[i][2]:.6f}\n")

        # use mtl
        f.write(f"g {object_name}_export\n")
        mtl_bump_name = "material_0"
        f.write(f"usemtl {mtl_bump_name}\n")

        # face
        f.write(f"# {faces.shape[0]} faces\n")
        faces += 1
        for i in range(faces.shape[0]):
            f.write(
                f"f {faces[i][0]}/{faces[i][0]} {faces[i][1]}/{faces[i][1]} {faces[i][2]}/{faces[i][2]}\n"
            )

        # vertex texture
        if uv is not None:
            for i in range(uv.shape[0]):
                f.write(f"vt {uv[i][0]:.6f} {uv[i][1]:.6f}\n")
        else:
            vertext = verts[:, 0:2] * 0 - 1
            # vertex position
            f.write(f"# {vertext.shape[0]} vertex texture\n")
            for i in range(vertext.shape[0]):
                f.write(f"vt {vertext[i][0]:.6f} {vertext[i][1]:.6f}\n")

    # generate MTL file
    if uv is not None:
        mtl_file = file_name.replace(".obj", ".mtl")
        Ka = [0.2, 0.2, 0.2]
        Kd = [1, 1, 1]
        Ks = [1, 1, 1]
        Tr = 1
        Ns = 0
        illum = 2
        with open(mtl_file, "a") as f:
            f.write("# write_obj (c) 2004 Gabriel Peyr\n")
            f.write(f"newmtl {mtl_bump_name}\n")
            f.write(f"Ka  {Ka[0]:.6f} {Ka[1]:.6f} {Ka[2]:.6f}\n")
            f.write(f"Kd  {Kd[0]:.6f} {Kd[1]:.6f} {Kd[2]:.6f}\n")
            f.write(f"Ks  {Ks[0]:.6f} {Ks[1]:.6f} {Ks[2]:.6f}\n")
            f.write(f"Tr  {Tr}\n")
            f.write(f"Ns  {Ns}\n")
            f.write(f"illum {illum}\n")
            f.write(f"map_Kd {texture_name}\n")
            f.write("#\n# EOF\n")


def create_colormap(verts):
    minx = verts[:, 0].min()
    miny = verts[:, 1].min()
    minz = verts[:, 2].min()
    maxx = verts[:, 0].max()
    maxy = verts[:, 1].max()
    maxz = verts[:, 2].max()
    r = (verts[:, 0] - minx) / (maxx - minx)
    g = (verts[:, 1] - miny) / (maxy - miny)
    b = (verts[:, 2] - minz) / (maxz - minz)
    colors = np.stack((r, g, b), axis=-1)
    assert colors.shape == verts.shape
    return colors


def get_deform_spec_coeff_transf(
    src_evecs, tar_evecs, tar_vert_mass, tar_verts, gt_p2p_tuple=None, fmap=None
):
    tar_coeff = get_proj_coeff(tar_evecs, tar_vert_mass, tar_verts)
    if fmap is not None:
        deform_vert = src_evecs @ fmap @ tar_coeff
        return deform_vert
    else:
        assert gt_p2p_tuple is not None
        gt_p2p_ar = np.array(gt_p2p_tuple)
        assert gt_p2p_ar.shape[1] == 2
        fmap = get_fmap_12_from_p2p(tar_evecs, src_evecs, gt_p2p_ar)
        deform_vert = src_evecs @ fmap @ tar_coeff
        return deform_vert


def get_fmap_12_from_p2p(evecs_src, evecs_tar, p2p_tuple):
    from scipy.linalg import lstsq

    return lstsq(evecs_tar[p2p_tuple[:, 1]], evecs_src[p2p_tuple[:, 0]])[0]  # (k1,k2)


def refine_p2p_21_via_fmap(evecs_src, evecs_tar, p2p_tuple, return_fmap=False):
    if isinstance(p2p_tuple, tuple):
        p2p_tuple = np.array(p2p_tuple)
    # Reverse so that we get Fmap in 1->2 direction
    # and P2P in 2->1 direction
    p2p_tuple_21 = np.c_[p2p_tuple[:, 1], p2p_tuple[:, 0]]
    fmap_12 = get_fmap_12_from_p2p(evecs_tar, evecs_src, p2p_tuple_21)
    p2p_21 = get_p2p_12_idx_from_fmap12(evecs_tar, evecs_src, fmap_12)
    if return_fmap:
        return p2p_21, fmap_12
    return p2p_21


def get_p2p_12_idx_from_fmap12(evecs_src, evecs_tar, fmap_12):
    """
    How to shuffle target points to match source points.
    """
    emb1 = evecs_src @ fmap_12.T
    emb2 = evecs_tar
    tar_tree = KDTree(emb1)
    p2p_12 = tar_tree.query(emb2, k=1, workers=-1)[1]
    return p2p_12.squeeze()


def auto_WKS(evals, evects, num_E, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
    sigma = 7 * (e_max - e_min) / num_E

    e_min += 2 * sigma
    e_max -= 2 * sigma

    energy_list = np.linspace(e_min, e_max, num_E)

    return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)


def get_all_operators_poisson(verts_list, faces_list, k_eig, op_cache_dir=None):
    N = len(verts_list)

    frames = [None] * N
    massvec = [None] * N
    L = [None] * N
    evals = [None] * N
    evecs = [None] * N
    gradX = [None] * N
    gradY = [None] * N

    face_basis = [None] * N
    grad_op_C = [None] * N
    div_op_C = [None] * N
    lap_op_0 = [None] * N
    lhs_mat = [None] * N

    # process in random order
    inds = [i for i in range(N)]
    random.shuffle(inds)

    for num, i in enumerate(inds):
        # print(
        #     "get_all_operators_poisson() processing {} / {} {:.3f}%".format(
        #         num, N, num / N * 100
        #     )
        # )
        outputs = get_operators_poisson(
            verts_list[i], faces_list[i], k_eig, op_cache_dir
        )
        frames[i] = outputs[0]
        massvec[i] = outputs[1]
        L[i] = outputs[2]
        evals[i] = outputs[3]
        evecs[i] = outputs[4]
        gradX[i] = outputs[5]
        gradY[i] = outputs[6]
        face_basis[i] = outputs[7]
        grad_op_C[i] = outputs[8]
        div_op_C[i] = outputs[9]
        lap_op_0[i] = outputs[10]
    return (
        frames,
        massvec,
        L,
        evals,
        evecs,
        gradX,
        gradY,
        face_basis,
        grad_op_C,
        div_op_C,
        lap_op_0,
        lhs_mat,
    )


def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(
        -np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2 * sigma**2)
    )  # (num_E,K)

    weighted_evects = evects[None, :, :] * coefs[:, None, :]  # (num_E,N,K)

    natural_WKS = np.einsum("tnk,nk->nt", weighted_evects, evects)  # (N,num_E)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1 / inv_scaling)[None, :] * natural_WKS
    else:
        return natural_WKS


def ensure_dir_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device("cpu")).numpy()


def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat


# Hash a list of numpy arrays
def hash_arrays(arrs):
    running_hash = hashlib.sha1()
    for arr in arrs:
        binarr = arr.view(np.uint8)
        running_hash.update(binarr)
    return running_hash.hexdigest()


def get_operators_poisson(
    verts,
    faces,
    k_eig=128,
    op_cache_dir=None,
    normals=None,
    overwrite_cache=False,
    truncate_cache=False,
):
    device = verts.device
    dtype = verts.dtype
    verts_np = toNP(verts)
    faces_np = toNP(faces)

    if np.isnan(verts_np).any():
        raise RuntimeError("tried to construct operators from NaN verts")

    found = False
    if op_cache_dir is not None:
        ensure_dir_exists(op_cache_dir)
        hash_key_str = str(hash_arrays((verts_np, faces_np)))

        i_cache_search = 0
        while True:
            # Form the name of the file to check
            search_path = os.path.join(
                op_cache_dir, hash_key_str + "_" + str(i_cache_search) + ".npz"
            )

            try:
                # print('loading path: ' + str(search_path))
                npzfile = np.load(search_path, allow_pickle=True)
                cache_verts = npzfile["verts"]
                cache_faces = npzfile["faces"]
                cache_k_eig = npzfile["k_eig"].item()

                # If the cache doesn't match, keep looking
                if (not np.array_equal(verts, cache_verts)) or (
                    not np.array_equal(faces, cache_faces)
                ):
                    i_cache_search += 1
                    print("hash collision! searching next.")
                    continue

                # print("  cache hit!")

                # If we're overwriting, or there aren't enough eigenvalues, just delete it; we'll create a new
                # entry below more eigenvalues
                if overwrite_cache:
                    print("  overwriting cache by request")
                    os.remove(search_path)
                    break

                if cache_k_eig < k_eig:
                    print("  overwriting cache --- not enough eigenvalues")
                    os.remove(search_path)
                    break

                if "L_data" not in npzfile:
                    print("  overwriting cache --- entries are absent")
                    os.remove(search_path)
                    break

                def read_sp_mat(prefix):
                    data = npzfile[prefix + "_data"]
                    indices = npzfile[prefix + "_indices"]
                    indptr = npzfile[prefix + "_indptr"]
                    shape = npzfile[prefix + "_shape"]
                    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
                    return mat

                # This entry matches! Return it.
                frames = npzfile["frames"]
                mass = npzfile["mass"]
                L = read_sp_mat("L")
                evals = npzfile["evals"][:k_eig]
                evecs = npzfile["evecs"][:, :k_eig]
                gradX = read_sp_mat("gradX")
                gradY = read_sp_mat("gradY")

                face_basis = np.zeros((faces_np.shape[0], 3, 3))
                grad_op_C = read_sp_mat("grad_op_C")
                div_op_C = read_sp_mat("div_op_C")
                lap_op_0 = read_sp_mat("lap_op_0")

                if truncate_cache and cache_k_eig > k_eig:
                    assert False, "Error, Case not covered"

                frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
                mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
                L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
                evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
                evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
                gradX = sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
                gradY = sparse_np_to_torch(gradY).to(device=device, dtype=dtype)

                # Poisson stuff
                grad_op_C = sparse_np_to_torch(grad_op_C).to(
                    device=device, dtype=torch.float64
                )
                div_op_C = sparse_np_to_torch(div_op_C).to(
                    device=device, dtype=torch.float64
                )
                lap_op_0 = sparse_np_to_torch(lap_op_0).to(device=device, dtype=dtype)
                # lhs_mat = sparse_np_to_torch(lhs_mat).to(device=device, dtype=dtype)
                found = True

                break

            except FileNotFoundError:
                print("  cache miss -- constructing operators")
                break

            except Exception as E:
                print("unexpected error loading file: " + str(E))
                print("-- constructing operators")
                break

    if not found:
        # No matching entry found; recompute.
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(
            verts, faces, k_eig, normals=normals
        )

        face_basis, grad_op_C, div_op_C, lap_op_0 = compute_poisson_operator(
            verts_np, faces_np
        )

        dtype_np = np.float32

        # Store it in the cache
        if op_cache_dir is not None:
            L_np = sparse_torch_to_np(L).astype(dtype_np)
            gradX_np = sparse_torch_to_np(gradX).astype(dtype_np)
            gradY_np = sparse_torch_to_np(gradY).astype(dtype_np)

            np.savez(
                search_path,
                verts=verts_np,
                frames=toNP(frames).astype(dtype_np),
                faces=faces_np,
                k_eig=k_eig,
                mass=toNP(mass).astype(dtype_np),
                L_data=L_np.data,
                L_indices=L_np.indices,
                L_indptr=L_np.indptr,
                L_shape=L_np.shape,
                evals=toNP(evals).astype(dtype_np),
                evecs=toNP(evecs).astype(dtype_np),
                gradX_data=gradX_np.data,
                gradX_indices=gradX_np.indices,
                gradX_indptr=gradX_np.indptr,
                gradX_shape=gradX_np.shape,
                gradY_data=gradY_np.data,
                gradY_indices=gradY_np.indices,
                gradY_indptr=gradY_np.indptr,
                gradY_shape=gradY_np.shape,
                # Sparse matrice, brace yourself..
                grad_op_C_data=grad_op_C.data,
                grad_op_C_indices=grad_op_C.indices,
                grad_op_C_indptr=grad_op_C.indptr,
                grad_op_C_shape=grad_op_C.shape,
                div_op_C_data=div_op_C.data,
                div_op_C_indices=div_op_C.indices,
                div_op_C_indptr=div_op_C.indptr,
                div_op_C_shape=div_op_C.shape,
                lap_op_0_data=lap_op_0.data,
                lap_op_0_indices=lap_op_0.indices,
                lap_op_0_indptr=lap_op_0.indptr,
                lap_op_0_shape=lap_op_0.shape,
            )
            face_basis = torch.from_numpy(face_basis).to(
                device="cpu", dtype=torch.float32
            )
            grad_op_C = sparse_np_to_torch(grad_op_C).to(
                device="cpu", dtype=torch.float32
            )
            div_op_C = sparse_np_to_torch(div_op_C).to(
                device="cpu", dtype=torch.float32
            )
            lap_op_0 = sparse_np_to_torch(lap_op_0).to(
                device="cpu", dtype=torch.float32
            )

    return (
        frames,
        mass,
        L,
        evals,
        evecs,
        gradX,
        gradY,
        face_basis,
        grad_op_C,
        div_op_C,
        lap_op_0,
    )


def compute_operators(verts, faces, k_eig, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    See get_operators() for a similar routine that wraps this one with a layer of caching.
    Torch in / torch out.
    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use
    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient
    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately,
    rather than as one complex sparse matrix.
    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix,
    like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    device = verts.device
    dtype = verts.dtype
    is_cloud = faces.numel() == 0

    eps = 1e-8

    verts_np = toNP(verts).astype(np.float64)
    faces_np = toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)

    # Build the scalar Laplacian
    if is_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(verts_np)
    else:
        L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
        massvec_np = pp3d.vertex_areas(verts_np, faces_np)
        massvec_np += eps * np.mean(massvec_np)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col
    # === Compute the eigenbasis
    if k_eig > 0:
        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise,
                # but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(
                    L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma
                )

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0.0, a_max=float("inf"))

                break
            except Exception as e:
                print(e)
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (
                    eps * 10**failcount
                )

    else:  # k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian. For point clouds, use a whole local neighborhood
    if is_cloud:
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edges = torch.tensor(
            np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype
        )
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(verts, edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)

    return frames, massvec, L, evals, evecs, gradX, gradY


def compute_poisson_operator(verts_np, faces_np):
    # face_basis, grad_C, Lap_0_inv, div_C,
    b_1, b_2, _ = igl.local_basis(verts_np, faces_np)
    face_basis = np.column_stack((b_1, b_2)).reshape(-1, 2, 3)
    grad_op_F = igl.grad(verts_np, faces_np)
    grad_op_C = Mat_F_to_C_sp(grad_op_F).tocsc()

    face_areas = igl.doublearea(verts_np, faces_np) / 2
    mass_faces_3C = diags(np.repeat(face_areas, 3)).tocsc()
    div_op_C = (grad_op_C.T @ mass_faces_3C).tocsc()
    lap_op_0 = div_op_C @ grad_op_C
    # lap_0_inv = spinv(lap_op_0 + 1e-6 * mass_verts).tocsc()
    # lhs_mat = lap_0_inv @ div_op_C

    return face_basis, grad_op_C, div_op_C, lap_op_0


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient.
    All values pointwise.
    - edges: (2, E)
    """

    edges_np = toNP(edges)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(N, N)
    ).tocsc()

    return mat


def build_tangent_frames(verts, faces, normals=None):
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals is None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where(
        (torch.abs(dot(vert_normals, basis_cand1)) < 0.9).unsqueeze(-1),
        basis_cand1,
        basis_cand2,
    )
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def vertex_normals(verts, faces, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    if faces.numel() == 0:  # point cloud
        _, neigh_inds = find_knn(
            verts, verts, n_neighbors_cloud, omit_diagonal=True, method="cpu_kd"
        )
        neigh_points = verts_np[neigh_inds, :]
        neigh_points = neigh_points - verts_np[:, np.newaxis, :]
        normals = neighborhood_normal(neigh_points)

    else:  # mesh
        normals = mesh_vertex_normals(verts_np, toNP(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, toNP(faces))

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask, :] = (
                np.random.RandomState(seed=777).rand(*verts.shape) - 0.5
            )[bad_normals_mask, :]
            normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)):
        raise ValueError("NaN normals :(")

    return normals


def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = toNP(
        face_normals(torch.tensor(verts), torch.tensor(faces))
    )  # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=-1, keepdims=True
    )

    return vertex_normals


def find_knn(
    points_source, points_target, k, largest=False, omit_diagonal=False, method="brute"
):
    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError(
            "omit_diagonal can only be used when source and target are same shape"
        )

    if method != "cpu_kd" and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = "cpu_kd"
        print("switching to cpu_kd knn")

    if method == "brute":
        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(
            -1, points_target.shape[0], -1
        )
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(
            points_source.shape[0], -1, -1
        )

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float("inf")

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == "cpu_kd":
        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape(
                (neighbors.shape[0], neighbors.shape[1] - 1)
            )

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def norm(x, highdim=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if len(x.shape) == 1:
        raise ValueError(
            "called normalize() on single vector of dim "
            + str(x.shape)
            + " are you sure?"
        )
    if not highdim and x.shape[-1] > 4:
        raise ValueError(
            "called normalize() with large last dimension "
            + str(x.shape)
            + " are you sure?"
        )
    return x / (norm(x, highdim=highdim) + divide_eps).unsqueeze(-1)


def face_coords(verts, faces):
    coords = verts[faces]
    return coords


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def edge_tangent_vectors(verts, frames, edges):
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad_point_cloud(verts, frames, n_neighbors_cloud=30):
    verts_np = toNP(verts)

    _, neigh_inds = find_knn(
        verts, verts, n_neighbors_cloud, omit_diagonal=True, method="cpu_kd"
    )

    # TODO this could easily be way faster. For instance we could avoid the weird edges format
    # and the corresponding pure-python loop via some numpy broadcasting of the same logic.
    # The way it works right now is just to share code with the mesh version. But its low priority since its preprocessing code.

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad(verts_np, torch.tensor(edges), edge_tangent_vecs)
