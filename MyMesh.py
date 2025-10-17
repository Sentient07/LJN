from collections import defaultdict
import math
from pathlib import Path

import igl
import numpy as np
import trimesh
from scipy.sparse import csc_matrix, csr_matrix, diags, lil_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.linalg import factorized
from tqdm import tqdm
from scipy.sparse.linalg import inv as sparse_inv
import scipy.sparse as sp
from mesh_utils import (
    Mat_F_to_C_sp,
    blockify_transpose,
    compute_cotangent,
    get_rotation_to_plane,
    polar_decomp_np,
    scale_to_unit_cube,
    scale_to_unit_sphere,
    vec_normalize,
    zero_center_pts,
)
from utils import get_mat_from_complex


class MyMesh(object):
    def __init__(self, path=None, verts=None, faces=None, scale_type="unit_sphere"):
        assert path is not None or (verts is not None and faces is not None)
        assert scale_type in ["unit_sphere", "unit_cube", "unit_area", None]
        self.scale_type = scale_type
        self.path = path
        self.name_str = Path(path).stem if path is not None else None
        if path is not None:
            self.read_and_scale_mesh()
        else:
            self.verts, self.faces = verts, faces
            self.scale_vertices()

        self.edges = igl.edges(self.faces)
        self.trigs = self.verts[self.faces]
        self.n_v = self.verts.shape[0]
        self.n_f = self.faces.shape[0]
        if not hasattr(self, "triMesh"):
            self.triMesh = trimesh.Trimesh(
                vertices=self.verts, faces=self.faces, process=False
            )

    def scale_vertices(self):
        if self.scale_type == "unit_sphere":
            self.verts = scale_to_unit_sphere(self.verts)
        elif self.scale_type == "unit_cube":
            self.verts = scale_to_unit_cube(self.verts)
        elif self.scale_type == "unit_area":
            # TODO: Avoid recomputation of mass matrix
            area = np.sqrt(
                np.sum(
                    igl.massmatrix(self.verts, self.faces, igl.MASSMATRIX_TYPE_VORONOI)
                )
            )
            self.verts = self.verts / area
        else:
            pass

    def read_and_scale_mesh(self):
        # v, f = igl.read_triangle_mesh(self.path)
        m_ = trimesh.load(self.path, process=False)
        v, f = np.array(m_.vertices), np.array(m_.faces)
        self.verts = v
        self.faces = f
        self.scale_vertices()

    def get_f_f_inc_mat(self):
        """
        (F,F) matrix with 1s at the faces incident to each face
        """
        n_f = self.faces.shape[0]
        col_idx = igl.triangle_triangle_adjacency(self.faces)[0].flatten()
        row_idx = np.arange(n_f).repeat(3)
        data = np.ones_like(col_idx)
        f_f_inc_mat = csr_matrix((data, (row_idx, col_idx)), shape=(n_f, n_f))
        return f_f_inc_mat

    def get_e_f_inc_ar(self):
        """
        (E,2) vector with faces incident to each edge
        """
        he_tree = KDTree(self.half_edges)
        fl_1 = he_tree.query(self.edges, workers=-1)[1]
        fl_2 = he_tree.query(self.edges[:, ::-1], workers=-1)[1]
        ef_inc_f = np.c_[fl_1 // 3, fl_2 // 3]
        return ef_inc_f

    def get_e_f_inc_mat(self):
        e_f_inc_ar = self.get_e_f_inc_ar()
        n_f = self.faces.shape[0]
        n_e = self.edges.shape[0]
        rows = e_f_inc_ar.flatten()
        cols = np.repeat(np.arange(n_e), 2)
        data = np.ones(len(rows))
        e2f = csr_matrix((data, (rows, cols)), shape=(n_f, n_e))
        return e2f

    def get_he_v_inc_mat(self):
        """
        (3F, V) matrix with 1s at the vertices incident to the half edge
        Multiplying with (V,k) vector will sum the vertex value to incident HE.
        """
        n_hv = self.half_edges.shape[0]
        rows = np.arange(n_hv).repeat(2)
        cols = self.half_edges.flatten()
        data = np.ones(len(rows))
        he_v_inc_mat = csr_matrix((data, (rows, cols)), shape=(n_hv, self.n_v))
        return he_v_inc_mat

    def get_he_f_inc_mat(self):
        n_he = self.half_edges.shape[0]
        n_f = self.faces.shape[0]
        assert n_he == n_f * 3
        row_idx = np.arange(n_f).repeat(3)
        col_idx = np.tile(np.arange(3), n_f) + np.arange(0, n_he, 3).repeat(3)
        data = np.ones(n_he)
        return csr_matrix((data, (row_idx, col_idx)), shape=(n_f, n_he))

    def get_v_he_inc_mat(self):
        """
        (V, 3F) matrix, transpose of get_he_v_inc_mat()
        Multiplying with (3F,k) vector will sum HE values at the vertex.
        """
        return self.get_he_v_inc_mat().T

    def get_v_oppHe_inc_mat(self):
        """
        (V, 3F) matrix with 1s at the half edge opposite to the vertex.
        Multiplying with (3F,k) vector will sum HE values at the vertex.
        """
        n_he = self.half_edges.shape[0]
        row_idx = np.arange(n_he)
        col_idx = self.faces.flatten()
        data = np.ones(n_he)
        return csr_matrix((data, (row_idx, col_idx)), shape=(n_he, self.n_v))

    def get_e_v_inc_mat(self):
        """
        (HE, V) matrix with 1s at the (source) vertex incident to the half edge.
        Multiplying with (3F, k) vector will sum *non-directional* HE values at vertices.

        Returns:
            scipy.sparse.csr_matrix: (V, 3F) matrix.
        """
        n_v = self.n_v
        n_he = self.half_edges.shape[0]
        row_idx = np.arange(n_he)
        col_idx = self.half_edges[:, 0]
        data = np.ones(n_he)
        edge_inc_mat = (
            coo_matrix((data, (row_idx, col_idx)), shape=(n_he, n_v)).tocsr().T
        )
        return edge_inc_mat

    def get_D0(self):
        """
        (3F, V) matrix
        """
        n_he = self.half_edges.shape[0]
        n_v = self.verts.shape[0]
        rows = np.arange(n_he).repeat(2)
        cols = self.half_edges.flatten()
        data = np.tile(np.array([-1, 1]), n_he)
        he2v = sp.csr_matrix((data, (rows, cols)), shape=(n_he, n_v))
        return he2v

    def get_one_form_grad_op_at_v(self):
        """
        (3F, 9V) matrix.
        Takes in a vector of shape (9V,) and sums the dot product with
        half-edge opposite to the vertex within each face.

        See Eqn.5 https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/
        """
        # blockify half-edges
        he_vecs = self.get_D0() @ self.verts
        he_bd = blockify_transpose(he_vecs)
        # to sum over faces, get incidence matrix
        he_f_inc_mat = self.get_he_f_inc_mat()
        # Because vectors are flattened, apply kron
        he_f_inc_mat_k = sp.kron(he_f_inc_mat, sp.eye(3))
        vector_grad_op = he_f_inc_mat_k @ he_bd
        v_oppHe_inc_mat = self.get_v_oppHe_inc_mat()
        v_oppHe_inc_k = sp.kron(v_oppHe_inc_mat, sp.eye(9))
        return vector_grad_op @ v_oppHe_inc_k

    def compute_integrability(self, one_form):
        # Get one form gradient operator
        vec_grad_op = self.get_one_form_grad_op_at_v()
        integ_val = vec_grad_op @ one_form.flatten()
        assert np.allclose(
            integ_val, np.zeros_like(integ_val)
        ), "Integrability not satisfied"

    @staticmethod
    def get_v_f_inc_mat(faces):
        """
        (V, F) matrix
        It's a static because tet also uses it
        """
        from scipy.sparse import csr_matrix

        n_v = np.max(faces) + 1
        n_f = faces.shape[0]

        # Prepare data for the sparse matrix
        rows = faces.flatten()  # Vertex indices
        cols = np.repeat(np.arange(n_f), faces.shape[1])  # Face indices
        data = np.ones(len(rows))  # Fill with 1s

        # Create a sparse matrix
        v2f = csr_matrix((data, (rows, cols)), shape=(n_v, n_f))

        return v2f

    def construct_tree(self):
        self.verts_tree = KDTree(self.verts)
        self.faces_tree = KDTree(self.trigs.reshape(-1, 3))
        self.half_edges_tree = KDTree(self.half_edges)
        self.edges_tree = KDTree(self.edges)

    def get_v_v_adj_mat(self):
        """
        (V, V) matrix
        """
        return igl.adjacency_matrix(self.faces)

    def get_v_v_inc_mat(self):
        """
        (V, V) *diagonal* matrix with 1/(degree of the vertex)
        """
        adj_mat = self.get_v_v_adj_mat()
        deg = np.asarray(np.sum(adj_mat, axis=1)).squeeze()
        deg_inv = 1.0 / deg
        return diags(deg_inv)

    def process_edge_attr(self, advanced=False):
        half_edges_ = igl.unique_edge_map(self.faces)[0]
        # So that it aligns with igl.cotmatrix_entries
        self.half_edges = MyMesh.fix_half_edge_order(half_edges_)
        if advanced:
            self.construct_tree()
            boundary_edges_uo = igl.exterior_edges(self.faces)
            # This is unoriented, so get opposite edges
            self.boundary_edges = np.r_[boundary_edges_uo, boundary_edges_uo[:, [1, 0]]]
            half_edge_set = set([tuple(i) for i in self.half_edges.tolist()])
            bnd_edge_set = set([tuple(i) for i in self.boundary_edges.tolist()])
            interior_edges_dup = np.array(list(half_edge_set - bnd_edge_set))
            # Remove duplicates
            self.interior_edges = np.unique(interior_edges_dup, axis=0)
            self.edge_tree = KDTree(self.edges)
            self.e_f_inc = self.get_e_f_inc_ar()
        # Construct face edges
        # Order is [1,2],[2,0],[0,1]
        face_edge_1 = self.faces[:, [1, 2]]
        face_edge_2 = self.faces[:, [2, 0]]
        face_edge_3 = self.faces[:, [0, 1]]
        # Stack them (F, 3, 3)
        self.face_edges = np.stack((face_edge_1, face_edge_2, face_edge_3), axis=1)
        self.vertex_normals = igl.per_vertex_normals(self.verts, self.faces)

    @staticmethod
    def fix_half_edge_order(he_igl):
        ### IGL has [[F,2],[F,2],[F,2]] format, we need [3F,2] format
        n_f = he_igl.shape[0] // 3
        return np.column_stack(
            (he_igl[:n_f], he_igl[n_f : 2 * n_f], he_igl[2 * n_f :])
        ).reshape(-1, 2)

    def get_surface_volume(self):
        self._check_water_tight()
        ## Implementation of http://multires.caltech.edu/pubs/ImplicitFairing.pdf,
        ## Equation 10., Assumes origin is inside the mesh.
        g_ = (self.verts[self.faces] ** 2).sum(axis=1) / 3  # (F, 3)
        edge_1 = self.verts[self.faces[:, 1]] - self.verts[self.faces[:, 0]]
        edge_2 = self.verts[self.faces[:, 2]] - self.verts[self.faces[:, 0]]
        edge_wedge = np.cross(edge_1, edge_2)  # (F, 3)
        edge_wedge = edge_wedge / np.linalg.norm(edge_wedge, axis=1, keepdims=True)
        vol_face = np.einsum("ij,ij->i", g_, edge_wedge) / 6.0
        return vol_face

    def get_surface_mesh_volume(self):
        return self.get_surface_volume().sum()

    def get_vertex_volume(self):
        v2f = MyMesh.get_v_f_inc_mat(self.faces)
        return v2f @ self.get_surface_volume()

    def process_diff_ops(self, fac_lap=True):
        self.process_edge_attr()
        # Frames
        b_1, b_2, b_n = igl.local_basis(self.verts, self.faces)
        self.face_b_1 = b_1  # (F, 3)
        self.face_b_2 = b_2  # (F, 3)
        self.face_normals = b_n  # (F, 3)
        self.face_basis = np.column_stack((self.face_b_1, self.face_b_2)).reshape(
            -1, 2, 3
        )
        self.face_frame = np.column_stack(
            (self.face_b_1, self.face_b_2, self.face_normals)
        ).reshape(-1, 3, 3)

        # Area elements
        self.vertex_mass = igl.massmatrix(
            self.verts, self.faces, igl.MASSMATRIX_TYPE_VORONOI
        )  # (V, V)
        self.face_areas = igl.doublearea(self.verts, self.faces) / 2.0  # (F,)
        self.face_mass = csc_matrix(diags(self.face_areas))  # (F,F)
        self.face_mass_3C = diags(np.repeat(np.abs(self.face_areas), 3))

        # Diff operators
        grad_op = igl.grad(self.verts, self.faces)  # (3F, V)
        self.grad_C = Mat_F_to_C_sp(grad_op).astype(np.float64)
        self.div_C = (self.grad_C.T @ self.face_mass_3C).astype(np.float64)
        self.lap_op_0 = ((self.div_C @ self.grad_C) + 1e-4 * self.vertex_mass).astype(
            np.float64
        )
        if fac_lap:
            self.lap_op_0_fac = factorized(self.lap_op_0.tocsr())

    def _check_edge_manifold(self):
        ### Doesn't consider boundary edges ###
        d_, _ = self.half_edges_tree.query(self.edges, k=2)
        return len(np.nonzero(d_[:, :2].max(axis=1) == 0.0)[0]) == 0

    def _check_water_tight(self):
        assert igl.exterior_edges(self.faces).shape[0] == 0, "Mesh is not watertight"

    def get_half_edge_face(self, v1_idx, v2_idx):
        """
        Get the face index of the half edge (v1_idx, v2_idx)
        """
        assert v1_idx != v2_idx
        d_, f_ = self.half_edges_tree.query((v1_idx, v2_idx))
        if d_ == 0.0:
            return f_ // 3
        return None

    def get_star_1_half_edge(self):
        """
        Mass matrix for (1-forms) defined on (dual) half-edges.
        Circumcentric dual/primal volume, done via cotanget of opposing angles.

        Returns:
            scipy.sparse.csr_matrix: (3F, 3F) diagonal matrix.
        """
        cot_entries = -2.0 * igl.cotmatrix_entries(self.verts, self.faces).reshape(-1)
        return sp.diags(cot_entries, format="csr")

    def get_star_1_edge(self):
        """
        Mass matrix for (1-forms) defined on edges.
        Circumcentric dual/primal volume, done via *sum of* cotanget of opposing angles.

        NOTE: While size is same as `get_star_1_half_edge`, the entries are different.
              The entries are *summed* over the half-edges, as done in cotan laplacian.
        Returns:
            scipy.sparse.csr_matrix: (3F, 3F) diagonal matrix.
        """
        row_indices, col_indices = self.half_edges[:, 1], self.half_edges[:, 0]
        cot_entries = np.asarray(self.lap_op_0[row_indices, col_indices])
        return sp.diags(cot_entries.flatten(), format="csr")

    def get_half_edge_divergence(self):
        """
        Divergence operator on half-edges (V, 3F) matrix.

        Returns:
            scipy.sparse.csr_matrix: (V, 3F) matrix.
        """
        return self.get_he_to_v_op() @ self.get_star_1_half_edge()

    def get_edge_divergence(self):
        """
        Divergence operator on edges (V, E) matrix.
        NOTE: Different from `get_half_edge_divergence`, here, we simply sum over edges (without sign).

        Returns:
            scipy.sparse.csr_matrix: (V, E) matrix.
        """
        edge_inc_mat = self.get_e_v_inc_mat()
        return edge_inc_mat @ self.get_star_1_edge()

    def get_exterior_derivative_0(self):
        """
        Exterior derivative operator on 0-forms (3F, V) matrix.

        Returns:
            scipy.sparse.csr_matrix: (3F, V) matrix.
        """
        return self.get_D0()

    def get_face_connection(self):
        # Borrowed from Guiallaume's mouette
        self.face_transports = {}
        for e in self.interior_edges:
            A, B = e
            pA, pB = self.verts[A], self.verts[B]
            # E = geom.Vec(pB-pA)
            e_1 = pB - pA

            # ASSUME: Edge manifold
            T1 = self.get_half_edge_face(A, B)
            T2 = self.get_half_edge_face(B, A)
            try:
                assert T1 is not None and T2 is not None  # Since interior edges
            except AssertionError:
                import pdb

                pdb.set_trace()
            X1, Y1 = self.face_basis[T1, 0, :], self.face_basis[T1, 1, :]
            X2, Y2 = self.face_basis[T2, 0, :], self.face_basis[T2, 1, :]
            angle1 = math.atan2(np.dot(e_1, Y1), np.dot(e_1, X1))
            angle2 = math.atan2(np.dot(e_1, Y2), np.dot(e_1, X2))
            self.face_transports[(T1, T2)] = angle1 - angle2
            self.face_transports[(T2, T1)] = angle2 - angle1

    def process_deformation_basis(
        self, n_ev=100, fix_signs=True, bending_weight=1e-4, remove_zeros=True
    ):
        """
        Processes eigenfunctions of hessian of the discrete shell operator.
        Eigenfunctions are the extrinsic vectorfields.
        Args:
            n_ev (int, optional): Number of eigenfunctions. Defaults to 100.
            fix_signs (bool, optional): Consistent sign of eigenvector. Defaults to True.
            bending_weight (float, optional): Bending weight. Defaults to 1e-4.
        """
        import pyshell

        print("Fixing signs? ", fix_signs)
        print("Removing zeros? ", remove_zeros)
        E, EMAP, EF, EI = igl.edge_flaps(self.faces)
        hess = pyshell.shell_deformed_hessian(
            self.verts, self.verts, self.faces, E, EMAP, EF, EI, bending_weight
        )
        vm_ = self.vertex_mass
        vert_mass_3C = sp.block_diag((vm_, vm_, vm_), format="lil")
        n_ev_ = n_ev + 6 if remove_zeros else n_ev

        v_evals, v_evecs = eigsh(hess, n_ev_, M=vert_mass_3C, sigma=0.0, which="LM")
        # Get the zero modes out
        if remove_zeros:
            v_evals, v_evecs = v_evals[6:], v_evecs[:, 6:]
        # Fix signs
        if fix_signs:
            ind = v_evecs[0, :] < 0
            v_evals[ind] *= -1
            v_evecs[:, ind] *= -1
        # Change the shape to (V, 3, 3)
        split_v = np.split(v_evecs, 3, axis=0)
        v_evecs_ = np.dstack(split_v).reshape(self.n_v, -1).reshape(self.n_v, n_ev, 3)
        # v_evecs_ = np.concatenate(split_v, axis=-1).reshape(-1, v_evecs.shape[1], 3)
        self.def_basis = v_evecs_
        self.def_evals = v_evals

    def process_elastic_basis(self, n_ev=100, fix_signs=True, bending_weight=1e-4):
        """
        Processes elastic basis. Compute shell eigenfunctions and project to normal.
        Args:
            n_ev (int, optional): Number of eigenfunctions. Defaults to 100.
            fix_signs (bool, optional): Consistent sign of eigenvector. Defaults to True.
            bending_weight (float, optional): Bending weight. Defaults to 1e-4.
        """
        if not hasattr(self, "def_basis"):
            self.process_deformation_basis(
                n_ev, fix_signs, bending_weight=bending_weight
            )
        if self.def_basis.shape[1] < n_ev:
            self.process_deformation_basis(
                n_ev, fix_signs, bending_weight=bending_weight
            )
        self.elastic_basis = np.einsum(
            "nij,nj->ni", self.def_basis[:, :n_ev], self.vertex_normals
        )

    def process_dec_ops(self):
        """
        Compute the exterior derivative operators and inner products
        """
        D_0 = lil_matrix((self.edges.shape[0], self.verts.shape[0]))
        D_1 = lil_matrix((self.faces.shape[0], self.edges.shape[0]))
        star_0_inv = sparse_inv(self.vertex_mass).tocsr()
        star_1 = lil_matrix((self.edges.shape[0], self.edges.shape[0]))
        for e_idx, (e_1, e_2) in enumerate(tqdm(self.edges)):
            D_0[e_idx, e_1] = -1
            D_0[e_idx, e_2] = 1
            T_1 = self.get_half_edge_face(e_1, e_2)
            T_2 = self.get_half_edge_face(e_2, e_1)
            if T_1 is not None and T_2 is not None:
                D_1[T_1, e_idx] = 1
                D_1[T_2, e_idx] = -1
                vert_f_idx_1 = np.setdiff1d(self.faces[T_1], [e_1, e_2])[0]
                vert_f_idx_2 = np.setdiff1d(self.faces[T_2], [e_2, e_1])[0]
                vec_1_t_1 = self.verts[vert_f_idx_1] - self.verts[e_1]
                vec_2_t_1 = self.verts[vert_f_idx_1] - self.verts[e_2]
                vec_1_t_2 = self.verts[vert_f_idx_2] - self.verts[e_1]
                vec_2_t_2 = self.verts[vert_f_idx_2] - self.verts[e_2]
                cot_1 = compute_cotangent(vec_1_t_1, vec_2_t_1)
                cot_2 = compute_cotangent(vec_1_t_2, vec_2_t_2)
                dual_by_primal = 0.5 * (cot_1 + cot_2)
                star_1[e_idx, e_idx] = dual_by_primal

        self.star_0_inv = star_0_inv
        self.star_1 = star_1.tocsr()
        self.star_1_inv = sparse_inv(star_1).tocsr()
        self.star_2 = diags(1 / self.face_areas).tocsr()
        self.D_0 = D_0.tocsr()
        self.D_1 = D_1.tocsr()
        # Compute Laplacians
        self.lap_0 = self.star_0_inv @ self.D_0.T @ self.star_1 @ self.D_0
        self.lap_1 = (
            self.star_1_inv @ D_1.T @ self.star_2 @ D_1
            + D_0 @ star_0_inv @ D_0.T @ star_1
        )
        self.lap_2 = self.D_1 @ self.star_1_inv @ self.D_1.T @ self.star_2

    def subtract_component_along_normal(self, quant):
        n_hat = self.vertex_normals
        # Compute the dot product between each vector in v and the normalized normal vector
        dot_products = np.einsum("nij,nj->ni", quant, n_hat)
        # Compute the projections
        projections = dot_products[:, :, np.newaxis] * n_hat[:, np.newaxis, :]
        # Subtract the projections from the original vectors
        resultant_matrices = quant - projections
        return resultant_matrices

    def project_jac_to_verts(self, Jac, extrinsic=True):
        jac_face = Jac.reshape(self.faces.shape[0], 3, 3)
        jac_face_1 = jac_face[:, :, 0]
        jac_face_2 = jac_face[:, :, 1]
        jac_face_3 = jac_face[:, :, 2]
        jac_vert_1 = igl.average_onto_vertices(self.verts, self.faces, jac_face_1)
        jac_vert_2 = igl.average_onto_vertices(self.verts, self.faces, jac_face_2)
        jac_vert_3 = igl.average_onto_vertices(self.verts, self.faces, jac_face_3)
        jac_vert = np.column_stack((jac_vert_1, jac_vert_2, jac_vert_3)).reshape(
            -1, 3, 3
        )
        if extrinsic:
            return jac_vert
        v_jac_tangent = self.subtract_component_along_normal(jac_vert)
        return v_jac_tangent.transpose(0, 2, 1)

    def process_lbo_basis(self, n_ev=200):
        import robust_laplacian

        L, M = robust_laplacian.mesh_laplacian(
            self.verts, self.faces, mollify_factor=1e-5
        )
        evals, evecs = eigsh(L, n_ev, M=M, sigma=-0.001)
        self.lbo_evals = evals
        self.lbo_evecs = evecs

    def get_face_mass_matrix(self):
        return self.face_mass_3C

    def get_intrinsic_jacobian(self, image_pts, return_proj=True):
        J_C = (self.grad_C @ image_pts).reshape(-1, 3, 3)
        J_C_proj = self.face_basis @ J_C
        if return_proj:
            return J_C_proj
        return J_C

    def get_extrinsic_frame(self, v, normalise=False):
        f = self.faces
        trig = v[f]
        trig_a, trig_b, trig_c = trig[:, 0, :], trig[:, 1, :], trig[:, 2, :]
        comp_1 = vec_normalize(trig_b - trig_a) if normalise else trig_b - trig_a
        comp_2 = vec_normalize(trig_c - trig_a) if normalise else trig_c - trig_a
        comp_3 = igl.local_basis(v, f)[-1]
        return np.column_stack((comp_1, comp_2, comp_3)).reshape(-1, 3, 3)

    def construct_fourth_vertex(self, verts=None, faces=None, dist_frac=1.0):
        if verts is None:
            verts = self.verts
        if faces is None:
            faces = self.faces
        trig = verts[faces]
        centroid = np.mean(trig, axis=1)
        trig_a, trig_b, trig_c = trig[:, 0, :], trig[:, 1, :], trig[:, 2, :]
        comp_1 = trig_b - trig_a
        comp_2 = trig_c - trig_a
        cross_pdt = np.cross(comp_1, comp_2)
        fourth_vert = centroid + cross_pdt / np.linalg.norm(
            cross_pdt, axis=1, keepdims=True
        )
        return fourth_vert

    def get_extrinsic_frame_lifted(self, v, f, trig_d):
        trig = v[f]
        trig_a, trig_b, trig_c = trig[:, 0, :], trig[:, 1, :], trig[:, 2, :]
        comp_1 = trig_b - trig_a
        comp_2 = trig_c - trig_a
        comp_3 = trig_d - trig_a
        return np.column_stack((comp_1, comp_2, comp_3)).reshape(-1, 3, 3)

    def get_extrinsic_frame_tet(self, tet_v, tet_f):
        trig_d = tet_v[tet_f[:, 3]]
        trig_a, trig_b, trig_c = (
            tet_v[tet_f[:, 0]],
            tet_v[tet_f[:, 1]],
            tet_v[tet_f[:, 2]],
        )
        comp_1 = trig_b - trig_a
        comp_2 = trig_c - trig_a
        comp_3 = trig_d - trig_a
        return np.column_stack((comp_1, comp_2, comp_3)).reshape(-1, 3, 3)

    def get_jacobian_from_image_frame(self, image_pts, ord="C", method="tet"):
        """
        Estimates the Jacobian between two meshes using the frame method.
        """
        assert self.verts.shape[0] == image_pts.shape[0]
        if method == "tet":
            src_tet = self.construct_fourth_vertex(self.verts, self.faces)
            src_frame = self.get_extrinsic_frame_lifted(self.verts, self.faces, src_tet)
            tar_tet = self.construct_fourth_vertex(image_pts, self.faces)
            tar_frame = self.get_extrinsic_frame_lifted(image_pts, self.faces, tar_tet)
            J_ext = np.linalg.inv(src_frame) @ tar_frame
        else:
            src_frame = self.get_extrinsic_frame(self.verts)
            tar_frame = self.get_extrinsic_frame(image_pts)
            J_ext = np.linalg.inv(src_frame) @ tar_frame

        return J_ext

    def get_tet_jacobian_from_image_frame(self, image_tet_v):
        """
        Estimates the Jacobian between two tet meshes using the frame method.
        """
        assert hasattr(self, "tet_faces")
        src_frame = self.get_extrinsic_frame_tet(self.tet_verts, self.tet_faces)
        tar_frame = self.get_extrinsic_frame_tet(image_tet_v, self.tet_faces)
        J_ext = np.linalg.inv(src_frame) @ tar_frame
        return J_ext

    def get_conformal_jacobian(self, image_pts):
        """
        Get closest conformal transformation for each triangle and return
        """
        src_verts = self.verts.copy()
        tar_verts = image_pts.copy()
        common_normal = np.array([0, 0, 1])
        source_normals = self.face_normals
        target_normals = igl.local_basis(image_pts, self.faces)[-1]
        src_rot_plane = get_rotation_to_plane(source_normals, common_normal).transpose(
            0, 2, 1
        )
        tar_rot_plane = get_rotation_to_plane(target_normals, common_normal).transpose(
            0, 2, 1
        )
        # Apply rotation to edges.
        src_f_e1 = (
            src_verts[self.face_edges[:, 0, 1]] - src_verts[self.face_edges[:, 0, 0]]
        )
        src_f_e2 = (
            src_verts[self.face_edges[:, 1, 1]] - src_verts[self.face_edges[:, 1, 0]]
        )
        tar_f_e1 = (
            tar_verts[self.face_edges[:, 0, 1]] - tar_verts[self.face_edges[:, 0, 0]]
        )
        tar_f_e2 = (
            tar_verts[self.face_edges[:, 1, 1]] - tar_verts[self.face_edges[:, 1, 0]]
        )
        src_frame = np.stack((src_f_e1, src_f_e2, source_normals), axis=1)
        tar_frame = np.stack((tar_f_e1, tar_f_e2, target_normals), axis=1)
        src_frame_rot = src_frame @ src_rot_plane
        tar_frame_rot = tar_frame @ tar_rot_plane
        transf_mat = np.linalg.inv(src_frame_rot) @ tar_frame_rot  # B matrix
        R_mat, Y_mat = polar_decomp_np(transf_mat)
        scale_fac = np.sqrt(np.linalg.det(Y_mat))
        return (
            scale_fac[:, None, None]
            * src_rot_plane
            @ R_mat
            @ np.linalg.inv(tar_rot_plane)
        )

    def deform_mesh_poission(self, J_C=None, proj_back=False):
        assert hasattr(self, "lap_op_0_fac")
        if proj_back:
            J_C = self.face_basis @ J_C
        rhs_mat = self.div_C @ (J_C.reshape(-1, 3))
        deformed = self.lap_op_0_fac(rhs_mat).reshape(-1, 3)
        deformed = zero_center_pts(deformed)
        return deformed

    def deform_tet_poission(self, J_C):
        assert hasattr(self, "tet_lap")
        rhs_mat = self.tet_div_op_C @ (J_C.reshape(-1, 3))
        deformed = self.tet_lap_fac(rhs_mat).reshape(-1, 3)
        deformed = zero_center_pts(deformed)
        return deformed

    def get_projection(self, pts):
        return igl.point_mesh_squared_distance(pts, self.verts, self.faces)[-1]

    def process_connec_Lap_greedy(self, n_conn_ev=200, is_complex=True, order=1):
        from mouette import mesh as mou_mesh
        from mouette.operators import laplacian_triangles
        from mouette.processing import SurfaceConnectionFaces

        assert self.path is not None
        m_mou = mou_mesh.load(self.path)
        # m_surf_conn = SurfaceConnectionFaces(m_mou)
        connec = SurfaceConnectionFaces(m_mou) if is_complex else None
        lap_mou = laplacian_triangles(m_mou, connection=connec, order=order)
        self.connec_lap = lap_mou
        mou_v = np.array(m_mou.vertices)
        mou_f = np.array(m_mou.faces)
        M = sp.diags(trimesh.triangles.area(mou_v[mou_f]), format="csc")
        self.connec_mass = M
        conn_b1, conn_b2, conn_b3 = igl.local_basis(mou_v, mou_f)
        self.connec_basis = np.column_stack((conn_b1, conn_b2)).reshape(-1, 2, 3)
        self.conn_evals, self.conn_evecs = eigsh(
            self.connec_lap, k=n_conn_ev, M=M, sigma=0.0
        )
        mou_b1, mou_b2 = connec._baseX, connec._baseY
        mou_basis = np.column_stack(
            (np.array(mou_b1._data), np.array(mou_b2._data))
        ).reshape(-1, 2, 3)
        self.mou_basis = mou_basis

    def process_connec_spectral(self, k):
        if not hasattr(self, "connec_lap"):
            self.process_connec_Lap_greedy()
        conn_eval, conn_evec = eigsh(self.connec_lap, k=k, which="LM", sigma=0.0)
        self.connec_evals = conn_eval
        self.connec_evecs = conn_evec
        self.connec_evecs_trans = (self.face_mass @ self.connec_evecs).T

    def barycentric_smooth(self, f_at_v):
        """Interpolate func at verts to centroids
        Args:
            f_at_v (np.ndarray): Function defined at vertices (#V,K)
        """
        f_at_f = f_at_v[self.faces]  # (#F, 3, K)
        f_at_f_centroid = f_at_f.mean(axis=1)  # (#F, K)
        return f_at_f_centroid

    def save_mesh(self, path, ext="obj"):
        path_w_ext = path + ext if "." not in Path(path).name else path
        _ = self.triMesh.export(path_w_ext)

    def get_vert_edge_inc_fac(self):
        if not hasattr(self, "v_e_incidence"):
            mat_A = np.zeros((3 * self.n_f, self.n_v + self.n_f))
            for f_iter, f_id in enumerate(self.faces):
                # For the vertex indicated by f_id[0], set -1 for all 3 components
                mat_A[3 * f_iter : 3 * f_iter + 3, f_id[0]] = -1
                mat_A[3 * f_iter, f_id[1]] = 1
                mat_A[3 * f_iter + 1, f_id[2]] = 1
                mat_A[3 * f_iter + 2, self.n_v + f_iter] = 1
            mat_A_sp = csc_matrix(mat_A)
            self.v_e_incidence = sp.kron(mat_A_sp, sp.eye(3))
            self.v_e_incidence_fac = factorized(
                self.v_e_incidence.T @ self.v_e_incidence
            )
        return self.v_e_incidence, self.v_e_incidence_fac

    def tetrahedralize(self, process_ops=False):
        self._check_water_tight()
        assert self._check_edge_manifold(), "Mesh is not edge manifold"

        def add_cover_verts(verts, faces):
            verts, faces = np.array(verts), np.array(faces)
            centroids = np.mean(verts[faces], axis=1)
            # mean edge length
            # mean_elen = igl.avg_edge_length(verts, faces)
            mean_elen = igl.avg_edge_length(verts, faces)
            v_1, v_2, v_3 = (
                self.trigs[:, 0, :],
                self.trigs[:, 1, :],
                self.trigs[:, 2, :],
            )
            cur_elen = (
                np.linalg.norm(v_1 - v_2, axis=1)
                + np.linalg.norm(v_2 - v_3, axis=1)
                + np.linalg.norm(v_3 - v_1, axis=1)
            )
            cur_elen = cur_elen / 3.0
            lifted_fv = (
                centroids + 0.1 * cur_elen[:, None] * igl.local_basis(verts, faces)[-1]
            )
            lifted_vv = verts + 0.1 * mean_elen * igl.per_vertex_normals(verts, faces)
            new_verts = np.vstack((verts, lifted_fv, lifted_vv))
            return new_verts

        def add_cover_faces(verts, faces, face_incidence_dict):
            lifted_fv_offset = verts.shape[0]
            lifted_vv_offset = verts.shape[0] + faces.shape[0]
            new_faces = []
            for f_ in range(faces.shape[0]):
                incident_faces = face_incidence_dict[f_]
                for i in range(len(incident_faces)):
                    # Sink should be the first vert
                    inc_f_idx, inc_e_idxs = tuple(incident_faces[i].items())[0]
                    src_idx, sink_idx = inc_e_idxs
                    new_faces.append(
                        [
                            sink_idx + lifted_vv_offset,
                            f_ + lifted_fv_offset,
                            inc_f_idx + lifted_fv_offset,
                        ]
                    )
                    new_faces.append(
                        [
                            f_ + lifted_fv_offset,
                            src_idx + lifted_vv_offset,
                            inc_f_idx + lifted_fv_offset,
                        ]
                    )
            new_faces = list({tuple(sorted(f)): f for f in new_faces}.values())
            new_faces = np.array(new_faces)
            return new_faces

        def get_face_incidence_dict(faces):
            face_incidence_dict = defaultdict(list)
            face_incidence_ar, opp_edge_idx = igl.triangle_triangle_adjacency(faces)
            for f in range(faces.shape[0]):
                for i in range(3):
                    inc_fv_idxs = faces[face_incidence_ar[f, i]]
                    inc_edges = [
                        (inc_fv_idxs[0], inc_fv_idxs[1]),
                        (inc_fv_idxs[1], inc_fv_idxs[2]),
                        (inc_fv_idxs[2], inc_fv_idxs[0]),
                    ]

                    inc_edge_opp = inc_edges[opp_edge_idx[f, i]]
                    face_incidence_dict[f].append(
                        {face_incidence_ar[f, i]: [inc_edge_opp[1], inc_edge_opp[0]]}
                    )
            return face_incidence_dict

        def add_thickening_faces(verts, faces):
            new_faces = []
            n_verts = verts.shape[0]
            for f_ in range(faces.shape[0]):
                new_faces.append([n_verts + f_, faces[f_, 0], faces[f_, 1]])
                new_faces.append([n_verts + f_, faces[f_, 1], faces[f_, 2]])
                new_faces.append([n_verts + f_, faces[f_, 2], faces[f_, 0]])

            # Not necessary probably? still...
            new_faces = list({tuple(sorted(f)): f for f in new_faces}.values())
            new_faces = np.array(new_faces)
            return new_faces

        def get_face_edge_dict(faces):
            # Between each face (Face_idx), get the edge idxs (v1_idx, v2_idx)
            # Key: tuple of face idxs, value: tuple of vertex idxs
            face_edge_dict = {}
            face_incidence_ar, opp_edge_idx = igl.triangle_triangle_adjacency(faces)
            for f_ in range(faces.shape[0]):
                for i in range(3):
                    inc_fv_idxs = faces[face_incidence_ar[f_, i]]
                    inc_edges = [
                        (inc_fv_idxs[0], inc_fv_idxs[1]),
                        (inc_fv_idxs[1], inc_fv_idxs[2]),
                        (inc_fv_idxs[2], inc_fv_idxs[0]),
                    ]
                    inc_edge_opp = sorted(inc_edges[opp_edge_idx[f_, i]])
                    face_edge_dict[(f_, face_incidence_ar[f_, i])] = tuple(inc_edge_opp)
            return face_edge_dict

        tet_verts = add_cover_verts(self.verts, self.faces)
        face_incidence_dict = get_face_incidence_dict(self.faces)
        cover_faces = add_cover_faces(self.verts, self.faces, face_incidence_dict)
        thicken_faces = add_thickening_faces(self.verts, self.faces)
        self.tet_cover_f = cover_faces
        self.tet_thicken_f = thicken_faces
        self.offset_tri_f = np.r_[cover_faces, thicken_faces]
        self.offset_tri_v = tet_verts[self.verts.shape[0] :,]
        # Map bw face idx and edge (vertex idxs) in between
        face_edge_dict = get_face_edge_dict(self.faces)
        edge_face_dict = {v: k for k, v in face_edge_dict.items()}
        # Tetrahedralize
        tet_faces = []
        lifted_fv_offset = self.verts.shape[0]
        lifted_vv_offset = self.verts.shape[0] + self.faces.shape[0]
        # 1st is upward tet - base are faces, apex is lifted centroid
        for f_ in range(self.faces.shape[0]):
            base_face = self.faces[f_]
            apex_idx = self.verts.shape[0] + f_
            tet_faces.append([base_face[0], base_face[1], base_face[2], apex_idx])

        # 2nd is downward tet - base are cover faces, apex is original vertex
        for f_ in range(cover_faces.shape[0]):
            cover_face_ = cover_faces[f_]
            # How to find apex? Simply subtract vv offset and positive is the apex (idx of orig vert)
            cover_face__ = cover_face_ - lifted_vv_offset
            apex_idx = cover_face__[cover_face__ >= 0]
            assert apex_idx.shape[0] == 1
            apex_idx = apex_idx[0]
            tet_faces.append([cover_face_[0], cover_face_[1], cover_face_[2], apex_idx])

        # 3rd is side tets - base are thicken faces, apex is original vertex
        for f_ in range(thicken_faces.shape[0]):
            thicken_face_ = thicken_faces[f_]
            # Apex is neighbour face's lifted centroid.
            # Subtract fv offset and positive value is current face idx.
            # Negative are edges. From that, get the neighbour face idx.
            thicken_face__ = thicken_face_ - lifted_fv_offset
            cur_face_idx = thicken_face__[thicken_face__ >= 0]
            assert cur_face_idx.shape[0] == 1
            base_edge_idx = thicken_face_[thicken_face__ < 0].tolist()
            cur_neigh_f_idx = list(edge_face_dict[tuple(sorted(tuple(base_edge_idx)))])
            assert cur_face_idx in cur_neigh_f_idx
            cur_neigh_f_idx.remove(cur_face_idx)
            tet_faces.append(
                [
                    thicken_face_[0],
                    thicken_face_[1],
                    thicken_face_[2],
                    cur_neigh_f_idx[0] + lifted_fv_offset,
                ]
            )

        tet_faces = np.array(tet_faces)

        self.tet_verts = tet_verts
        self.tet_faces = tet_faces
        if process_ops:
            tet_vol = igl.volume(tet_verts, tet_faces.astype(np.int32))
            self.tet_vol = tet_vol
            tet_grad = igl.grad(tet_verts, tet_faces.astype(np.int32))
            self.tet_grad_C = Mat_F_to_C_sp(tet_grad)
            # It's repeat because C-ordering
            self.tet_vol_diag_C = diags(np.repeat(np.abs(tet_vol), 3))
            self.tet_div_op_C = self.tet_grad_C.T @ self.tet_vol_diag_C
            self.tet_lap = self.tet_div_op_C @ self.tet_grad_C
            # get_vertex_face_incidence
            tet_fv_inc = MyMesh.get_v_f_inc_mat(self.tet_faces)
            self.tet_mass = diags((tet_fv_inc @ tet_vol) / 4)
            self.tet_lap_fac = factorized(self.tet_lap + self.tet_mass * 1e-6)

    def set_geod_mat(self):
        import potpourri3d as pp3d

        print("Setting Geodesic matrix bw vertices")
        n_vertices = self.verts.shape[0]
        distmat = np.zeros((n_vertices, n_vertices))
        solver = pp3d.MeshHeatMethodDistanceSolver(self.verts, self.faces)
        iterable = tqdm(range(n_vertices))
        for vertind in iterable:
            distmat[vertind] = np.maximum(solver.compute_distance(vertind), 0)
        self.geod_mat = distmat

    def process_cross_fields(self):
        if not hasattr(self, "connec_lap"):
            self.process_connec_Lap_greedy(n_conn_ev=5, is_complex=True)
        cross_x = self.conn_evecs[:, 0]
        cross_x = cross_x / np.abs(cross_x)  # normalize
        # multiply by \pi/2 to get the cross field
        cross_y = cross_x * 1j
        cross_complex = np.column_stack((cross_x, cross_y))
        cross_real = get_mat_from_complex(cross_complex)
        cross_real_vec = np.einsum("fkc,fcd->fkd", cross_real, self.connec_basis)
        self.cross_complex = cross_complex
        self.cross_real = cross_real_vec

    def get_he_to_v_op(self):
        """
        Signed incidence matrix (V,3F)
        Sum the half-edges around each vertex with the sign of the half-edge
        and divide by 2
        Args:
            he : half-edges
            n_v : number of vertices

        Returns:
            S: (n_v, n_he) sparse matrix
        """
        half_edges = self.half_edges
        n_v = self.n_v
        n_he = len(half_edges)
        rows = np.concatenate([half_edges[:, 0], half_edges[:, 1]])
        cols = np.concatenate([np.arange(len(half_edges)), np.arange(len(half_edges))])
        data = np.concatenate(
            [np.full(len(half_edges), 0.5), np.full(len(half_edges), -0.5)]
        )
        S_coo = coo_matrix((data, (rows, cols)), shape=(n_v, n_he), dtype=np.float32)
        return S_coo.tocsr()

    def get_e_to_v_inc(self):
        """
        Unsigned incidence matrix (V, E)
        """
        n_e = self.edges.shape[0]
        edges = self.edges
        n_v = self.verts.shape[0]
        # Initialize an empty sparse matrix
        rows = (
            edges.flatten()
        )  # Flatten edge pairs to get repeating row indices [i, j, i, j, ...]
        cols = np.repeat(
            np.arange(n_e), 2
        )  # Repeat each column index twice, once for each vertex in the edge
        data = np.ones_like(
            rows, dtype=np.float32
        )  # Initialize the data to 1 for each non-zero entry
        S_coo = coo_matrix((data, (rows, cols)), shape=(n_v, n_e), dtype=np.float32)
        return S_coo.tocsr()

    def get_v_to_f_inc(self, weight="equal"):
        # Unsigned incidence matrix (V, F)
        if weight != "equal":
            raise NotImplementedError
        n_f = self.faces.shape[0]
        n_v = self.verts.shape[0]
        # Initialize an empty sparse matrix
        rows = np.repeat(np.arange(n_f), 3)
        cols = self.faces.flatten()
        data = np.ones_like(rows, dtype=np.float32) * 1 / 3
        S_coo = coo_matrix((data, (rows, cols)), shape=(n_f, n_v), dtype=np.float32)
        return S_coo.tocsr()

    def get_f_to_v_inc(self, weight="equal"):
        # Unsigned incidence matrix (F, V)
        if weight != "equal":
            raise NotImplementedError
        S = MyMesh.get_v_f_inc_mat(self.faces)
        face_inc = np.bincount(self.faces.flatten())
        face_inc = 1.0 / face_inc
        inc_diag = sp.diags(face_inc, format="csr")
        return inc_diag @ S
