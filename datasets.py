# datasets.py

from collections import defaultdict
from glob import glob
from pathlib import Path
import os
import os.path as osp
from itertools import permutations, combinations

import igl
import numpy as np
import point_cloud_utils as pcu
import torch
from scipy.sparse import coo_matrix, eye as speye
from scipy.sparse import diags as sp_diags
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import (
    get_all_operators_poisson,
    auto_WKS,
    numpied,
    get_jac_at_vert,
    polar_decomp_np,
    sparse_np_to_torch,
    read_map,
    read_vts,
    get_p2p_tup_from_vts,
    refine_p2p_21_via_fmap,
    get_deform_spec_coeff_transf,
    get_only_rotation_np,
    sparse_np_to_tsp,
)
import DATA_PATHS as d_p
from MyMesh import MyMesh


def torched_(np_ar, dtype=torch.float32):
    return torch.from_numpy(np_ar).to(dtype).to("cpu")


class BatchJacDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        names_list=None,
        all_shapes=None,
        full_verts=False,
        mesh_ext="obj",
        is_permut=True,
        init_ops=True,
        init_jac=True,
        init_combo=True,
    ) -> None:
        """
        Dataset for loading mesh pairs and their precomputed data for Jacobian-related tasks.

        Args:
            data_dir (str, optional): Directory containing mesh files. Defaults to None.
            names_list (list, optional): List of mesh names (without extension). Defaults to None.
            all_shapes (list, optional): List of full paths to mesh files. Overrides data_dir and names_list. Defaults to None.
            full_verts (bool, optional): If True, use all vertices. If False, sample points on the surface. Defaults to False.
            mesh_ext (str, optional): Mesh file extension. Defaults to 'obj'.
            is_permut (bool, optional): If True, create all permutations of pairs. If False, create combinations. Defaults to True.
            init_ops (bool, optional): If True, precompute differential operators. Defaults to True.
            init_jac (bool, optional): If True, precompute ground truth Jacobians. Defaults to True.
            init_combo (bool, optional): If True, initialize mesh pair combinations. Defaults to True.
        """
        if all_shapes is None:
            all_shapes = [
                osp.join(data_dir, "%s.%s" % (i, mesh_ext)) for i in names_list
            ]

        self.all_shapes = all_shapes
        self.all_meshes = [MyMesh(i) for i in tqdm(all_shapes)]
        self.sel_meshes = self.all_meshes
        self.is_permut = is_permut
        self.full_verts = full_verts
        if init_combo:
            self.init_combo_idx()
        if init_ops:
            self.init_all_ops()
        if init_jac:
            self.init_gt_jac()

    def __len__(self):
        return len(self.combo_idx)

    def init_all_ops(self) -> None:
        """Precomputes differential operators and other properties for all meshes."""
        for i in tqdm(self.all_meshes, desc="Processing Meshes"):
            i.process_diff_ops(fac_lap=True)
            i.frame_inv = np.linalg.inv(
                i.get_extrinsic_frame(i.verts).astype(np.float64)
            )
            cot_entries = -2.0 * igl.cotmatrix_entries(i.verts, i.faces).reshape(-1)
            i.he_to_v_op = (i.get_he_to_v_op() @ sp_diags(cot_entries)).astype(
                np.float64
            )
            i.e_f_inc_ar = i.get_e_f_inc_ar()
            # The following seem to be placeholders and are not used in __getitem__
            i.v_f_avg_op = speye(3).tocoo()
            i.f_inc = speye(3).tocoo()

    def init_combo_idx(self) -> None:
        """Initializes the list of mesh pairs (combinations or permutations)."""
        if self.is_permut:
            combo_idx = permutations(range(len(self.all_meshes)), 2)
        else:
            combo_idx = combinations(range(len(self.all_meshes)), 2)

        self.combo_idx = list(combo_idx)

    def sample_surface(
        self, my_mesh: MyMesh, num_samp: int = 2000
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Samples points and their corresponding normals from the mesh surface.

        Args:
            my_mesh (MyMesh): The mesh to sample from.
            num_samp (int, optional): Number of points to sample. Defaults to 2000.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing sampled points and normals.
        """
        v, f, n = my_mesh.verts, my_mesh.faces, my_mesh.vertex_normals
        f_i, bc = pcu.sample_mesh_random(v, f, num_samp)
        v_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
        n_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, n)
        return v_poisson, n_poisson

    def init_gt_jac(self) -> None:
        """Computes and caches the ground truth Jacobians for all mesh pairs."""
        all_jac = {}
        for i, j in tqdm(self.combo_idx, desc="Computing GT Jacobians"):
            src_m, tar_m = self.sel_meshes[i], self.sel_meshes[j]
            if osp.isfile(
                osp.join(
                    self.cache_dir, src_m.name_str + "_" + tar_m.name_str + "_jac.npz"
                )
            ):
                gt_jac = np.load(
                    osp.join(
                        self.cache_dir,
                        src_m.name_str + "_" + tar_m.name_str + "_jac.npz",
                    )
                )["gt_jac"]
                all_jac[src_m.name_str + "_" + tar_m.name_str] = gt_jac
                continue
            gt_jac = src_m.get_jacobian_from_image_frame(tar_m.verts)
            all_jac[src_m.name_str + "_" + tar_m.name_str] = gt_jac
            if self.cache_dir:
                np.savez(
                    osp.join(
                        self.cache_dir, f"{src_m.name_str}_{tar_m.name_str}_jac.npz"
                    ),
                    gt_jac=gt_jac,
                )
        self.all_gt_jac = all_jac

    def __getitem__(self, idx):
        src_idx, tar_idx = self.combo_idx[idx]
        src_m, tar_m = self.all_meshes[src_idx], self.all_meshes[tar_idx]
        src_cent = torched_(np.mean(src_m.verts[src_m.faces], axis=1))

        if self.full_verts:
            tar_p, tar_n = tar_m.verts, tar_m.vertex_normals
        else:
            tar_p, tar_n = self.sample_surface(tar_m, 2000)
            src_p, src_n = self.sample_surface(src_m, 2000)
            src_p, src_n = torched_(src_p), torched_(src_n)
            tar_p, tar_n = torched_(tar_p), torched_(tar_n)

        src_face_basis = torched_(src_m.face_basis)
        src_div = src_m.div_C.astype(np.float64).tocoo()

        gt_jac = self.all_gt_jac[f"{src_m.name_str}_{tar_m.name_str}"]
        gt_jac = torched_(gt_jac.reshape(-1, 3, 3))

        src_f_th = torched_(src_m.faces, dtype=torch.int64)
        src_v_th = torched_(src_m.verts)
        tar_v_th = torched_(tar_m.verts)
        src_lap = (src_m.lap_op_0 + 1e-4 * src_m.vertex_mass).tocoo()
        src_fn = torched_(igl.local_basis(src_m.verts, src_m.faces)[-1])
        frame_inv = torched_(src_m.frame_inv)
        src_he_idx = src_m.half_edges
        src_he_vec = src_v_th[src_he_idx[:, 1]] - src_v_th[src_he_idx[:, 0]]
        he_to_v_op = src_m.he_to_v_op.tocoo()
        e_f_inc_ar = torched_(src_m.e_f_inc_ar).long()
        edges = torched_(src_m.edges).long()

        combo_idx = str(src_idx) + "_" + str(tar_idx)
        return (
            src_v_th,
            src_f_th,
            src_cent,
            src_p,
            src_he_vec,
            he_to_v_op,
            src_face_basis,
            src_fn,
            src_div,
            src_lap,
            frame_inv,
            e_f_inc_ar,
            edges,
            tar_p,
            tar_n,
            tar_v_th,
            gt_jac,
            combo_idx,
        )


class DfnBatchJacDataset(BatchJacDataset):
    def __init__(
        self,
        data_dir=None,
        names_list=None,
        all_shapes=None,
        full_verts=False,
        mesh_ext="obj",
        is_permut=True,
        n_eig_proc=128,
        cache_dir=None,
        spec_inp=False,
        init_fm=True,
        init_jac=True,
        init_combo=True,
        init_ops=True,
        init_cache=True,
        varying_fm=False,
        rot_inp=False,
        one_one=True,
    ):
        """
        Dataset for DiffusionNet-based models, extending BatchJacDataset.

        Args:
            n_eig_proc (int, optional): Number of eigenvalues/eigenvectors to compute. Defaults to 128.
            cache_dir (str, optional): Directory to cache precomputed operators. Defaults to None.
            spec_inp (bool, optional): Whether to use spectral input. Defaults to False.
            init_fm (bool, optional): If True, initialize functional maps. Defaults to True.
            init_jac (bool, optional): If True, initialize ground truth Jacobians. Defaults to True.
            init_combo (bool, optional): If True, initialize mesh pair combinations. Defaults to True.
            init_ops (bool, optional): If True, initialize differential operators. Defaults to True.
            init_cache (bool, optional): If True, initialize and use cache for operators. Defaults to True.
            varying_fm (bool, optional): If True, use a varying number of eigenvectors for functional maps. Defaults to False.
            rot_inp (bool, optional): If True, use only the rotational part of the Jacobian as input. Defaults to False.
            one_one (bool, optional): If True, assumes one-to-one correspondence for smooth Jacobian calculation. Defaults to True.
        """
        if cache_dir is None and data_dir is not None:
            self.cache_dir = osp.join(data_dir, "cache")
        else:
            self.cache_dir = None
        super().__init__(
            data_dir=data_dir,
            names_list=names_list,
            all_shapes=all_shapes,
            full_verts=full_verts,
            mesh_ext=mesh_ext,
            is_permut=is_permut,
            init_ops=init_ops,
            init_jac=init_jac,
            init_combo=init_combo,
        )
        self.n_eig_proc = n_eig_proc
        self.rot_inp = rot_inp
        print(f"Using only rot input: {self.rot_inp}")
        self.varying_fm = varying_fm
        self.spec_inp = spec_inp
        self.one_one = one_one
        if init_cache:
            self._init_cache()
        if init_fm:
            self._init_fmap()

    def _init_cache(self) -> None:
        """Initializes and caches DiffusionNet operators for all meshes."""
        self.all_ops = {}
        for my_m in tqdm(self.sel_meshes, desc="Computing Operators"):
            cur_v, cur_f = my_m.verts, my_m.faces
            cache_dir_ = (
                self.cache_dir
                if self.cache_dir is not None
                else osp.join(Path(my_m.name_str).parent, "cache")
            )
            ops_ = get_all_operators_poisson(
                [torched_(cur_v)],
                [torched_(cur_f).long()],
                k_eig=self.n_eig_proc,
                op_cache_dir=cache_dir_,
            )
            ops_ = list(ops_)
            # compute WKS
            wks = auto_WKS(
                numpied(ops_[3][0].squeeze()), numpied(ops_[4][0].squeeze()), 128
            )
            ops_.append(torched_(wks).unsqueeze(0))
            evec = ops_[4][0].double().squeeze()
            evec_pinv = torch.linalg.pinv(evec)
            ops_.append(evec_pinv.unsqueeze(0))
            self.all_ops[my_m.name_str] = tuple(ops_)
            spec_embed = numpied(evec[:, :40]) @ numpied(evec_pinv[:40, :]) @ cur_v
            my_m.spec_embed = spec_embed
            my_m.spec_frame = my_m.get_extrinsic_frame(spec_embed)

    def _init_fmap(self) -> None:
        """Initializes functional maps and related data for non one-to-one correspondences."""
        self.all_pulled_v = {}  # This seems unused, but keeping for potential subclasses.
        self.all_sm_jac = {}
        for i, j in tqdm(self.combo_idx, desc="Computing FMAP"):
            if self.varying_fm:
                if np.random.uniform() < 0.95:
                    n_ev = np.random.randint(30, 80)
                else:
                    n_ev = np.random.randint(100, 200)
            else:
                n_ev = 40
            src_m, tar_m = self.sel_meshes[i], self.sel_meshes[j]
            tar_evecs = numpied(self.all_ops[tar_m.name_str][4][0])[:, :n_ev]
            tar_evecs_pinv = numpied(self.all_ops[tar_m.name_str][-1][0])[:n_ev, :]
            pulled_v = tar_evecs @ tar_evecs_pinv @ tar_m.verts
            self.all_pulled_v[str(i) + "_" + str(j)] = pulled_v
            sm_jac = src_m.frame_inv @ src_m.get_extrinsic_frame(pulled_v)
            self.all_sm_jac[str(i) + "_" + str(j)] = sm_jac
            src_m.spec_embed = pulled_v

    def __getitem__(self, idx):
        src_idx, tar_idx = self.combo_idx[idx]
        if isinstance(src_idx, int) and isinstance(tar_idx, int):
            src_m, tar_m = self.sel_meshes[src_idx], self.sel_meshes[tar_idx]
        else:
            src_m, tar_m = self.mesh_name_dict[src_idx], self.mesh_name_dict[tar_idx]

        src_op_list = self.all_ops[src_m.name_str]
        (
            _,
            src_mass,
            src_lap,
            src_evals,
            src_evecs,
            src_gradX,
            src_gradY,
            _,
            _,
            _,
            _,
            _,
            src_wks,
            src_evecs_pinv,
        ) = [i[0] for i in src_op_list]

        tar_evecs = numpied(self.all_ops[tar_m.name_str][4][0])
        tar_mass = self.all_ops[tar_m.name_str][1][0]
        src_face_basis = torched_(src_m.face_basis)
        src_div = src_m.div_C.astype(np.float64).tocoo()

        gt_jac = self.all_gt_jac[src_m.name_str + "_" + tar_m.name_str]

        src_f_th = torched_(src_m.faces, dtype=torch.int64)
        tar_f_th = torched_(tar_m.faces, dtype=torch.int64)
        src_v_th = torched_(src_m.verts)
        tar_v_th = torched_(tar_m.verts)
        src_lap_op = (src_m.lap_op_0).tocoo() + 1e-2 * igl.massmatrix(
            src_m.verts, src_m.faces
        )

        pair_name = (src_m.name_str, tar_m.name_str)
        if hasattr(self, "all_p2p_map"):
            p2p_map = torched_(
                self.all_p2p_map[str(src_idx) + "_" + str(tar_idx)]
            ).long()
        else:
            p2p_map = torch.arange(src_m.verts.shape[0]).long()

        pulled_v = tar_m.spec_embed

        if self.one_one or not hasattr(self, "all_sm_jac"):
            sm_jac = src_m.frame_inv @ tar_m.spec_frame
        else:
            sm_jac = self.all_sm_jac.get(
                f"{src_idx}_{tar_idx}",
                np.eye(3)[np.newaxis, :, :].repeat(len(src_m.faces), axis=0),
            )

        frame_inv = torched_(src_m.frame_inv)
        if not self.rot_inp:
            sm_J_V = torched_(get_jac_at_vert(src_m.verts, src_m.faces, sm_jac))
        else:
            J_V = get_jac_at_vert(src_m.verts, src_m.faces, gt_jac)
            sm_J_V = torched_(polar_decomp_np(J_V)[0])
        sm_jac = torched_(sm_jac)
        pulled_v = torched_(pulled_v)
        tar_evecs = torched_(tar_evecs)
        gt_jac = torched_(gt_jac.reshape(-1, 3, 3))
        src_lap_op = sparse_np_to_torch(src_lap_op)
        src_div = sparse_np_to_torch(src_div)
        ret_items = (
            src_v_th,
            src_f_th,
            tar_v_th,
            tar_f_th,
            tar_evecs,
            tar_mass,
            src_face_basis,
            src_div,
            src_lap_op,
            gt_jac,
            p2p_map,
            sm_jac,
            sm_J_V,
            frame_inv,
            src_mass,
            src_lap,
            src_evals,
            src_evecs,
            src_gradX,
            src_gradY,
            pulled_v,
            pair_name,
        )
        return ret_items


class RemeshedDfnDataset(DfnBatchJacDataset):
    def __init__(
        self,
        data_name,
        data_dir=None,
        names_list=None,
        all_shapes=None,
        corr_dir=None,
        full_verts=False,
        mesh_ext="obj",
        is_permut=True,
        n_eig_proc=128,
        cache_dir=None,
        spec_inp=False,
        n_test_ev=20,
    ):
        """
        Dataset for remeshed shapes with known correspondences.

        Args:
            data_name (str): Name of the dataset (e.g., 'shrec19_r').
            data_dir (str, optional): Directory of mesh data. Defaults to None.
            names_list (list, optional): List of shape names. Defaults to None.
            all_shapes (list, optional): List of full paths to shapes. Defaults to None.
            corr_dir (str, optional): Directory of correspondence files. Defaults to None.
            full_verts (bool, optional): Use full vertices or sample. Defaults to False.
            mesh_ext (str, optional): Mesh file extension. Defaults to 'obj'.
            is_permut (bool, optional): Use permutations for pairs. Defaults to True.
            n_eig_proc (int, optional): Number of eigenvectors for operators. Defaults to 128.
            cache_dir (str, optional): Cache directory. Defaults to None.
            spec_inp (bool, optional): Use spectral input. Defaults to False.
            n_test_ev (int, optional): Number of eigenvectors for test-time functional map refinement. Defaults to 20.
        """
        print(f"Loading remeshed dataset: {data_name}")
        self.data_name = data_name
        self.cache_dir = cache_dir
        self.corr_dir = corr_dir
        self.n_test_ev = n_test_ev
        init_combo = True
        self.names_list = names_list
        if "shrec" in data_name.lower() or "dt4dh" in data_name.lower():
            init_combo = False  # Combinations are initialized manually
        super().__init__(
            data_dir=data_dir,
            names_list=names_list,
            all_shapes=all_shapes,
            full_verts=full_verts,
            mesh_ext=mesh_ext,
            is_permut=is_permut,
            n_eig_proc=n_eig_proc,
            cache_dir=cache_dir,
            spec_inp=spec_inp,
            init_fm=False,
            init_jac=False,
            init_combo=init_combo,
        )
        self.one_one = False
        if "shrec" in data_name.lower():
            self.init_shrec_combo()

        if "dt4dh" in data_name.lower():
            self.init_dt4d_combo()

        self.init_gt_jac()
        self._init_fmap()

    def init_shrec_combo(self) -> None:
        """Initializes pairs for the SHREC'19 dataset."""
        if self.names_list is None:
            combo_idxs = [
                Path(i).stem.split("_")
                for i in os.listdir(d_p.shrec19_r_corr)
                if i.endswith(".map")
            ]
            combo_idxs = [i for i in combo_idxs if int(i[0]) != 40 and int(i[1]) != 40]
            self.combo_idx = [(int(i[0]) - 1, int(i[1]) - 1) for i in combo_idxs]
        else:
            self.combo_idx = [(1, 0)]

    def init_dt4d_combo(self) -> None:
        """Initializes pairs for the DeformingThings4D dataset."""
        n_meshes = len(self.all_meshes)
        # Create pairs between the two halves of the dataset
        self.combo_idx = np.c_[
            np.arange(n_meshes // 2), np.arange(n_meshes // 2, n_meshes)
        ].tolist()

    def init_gt_jac(self) -> None:
        """Initializes ground truth Jacobians as identity for remeshed datasets (no deformation)."""
        self.all_gt_jac = {}
        for i, j in tqdm(self.combo_idx, desc="Computing GT Jacobians"):
            src_m, tar_m = self.sel_meshes[i], self.sel_meshes[j]
            n_f = len(src_m.faces)
            self.all_gt_jac[src_m.name_str + "_" + tar_m.name_str] = np.eye(3)[
                np.newaxis, :, :
            ].repeat(n_f, axis=0)

    def _init_fmap(self) -> None:
        """Initializes functional maps and pulls back vertex positions for remeshed datasets."""
        self.all_p2p_map = {}
        self.all_f_map = {}
        self.all_pulled_v = {}
        self.all_arap_jac_int = {}
        self.all_sm_jac = {}
        print("Initing non 1-1 FM")
        for i, j in tqdm(self.combo_idx, desc="Computing FMAP"):
            src_name = self.sel_meshes[i].name_str
            tar_name = self.sel_meshes[j].name_str
            src_m, tar_m = self.sel_meshes[i], self.sel_meshes[j]
            if "shrec" in self.data_name.lower():
                p2p_ar = read_map(d_p.shrec19_r_corr, "%s_%s" % (src_name, tar_name))
                p2p_tuple = np.c_[np.arange(len(src_m.verts)), p2p_ar]
            elif "dt4dh" in self.data_name.lower():
                src_cat, tar_cat = (
                    self.data_name.split("_")[2],
                    self.data_name.split("_")[3],
                )
                cross_cat_map = osp.join(
                    d_p.dt4d_r_corr, "%s_%s.vts" % (src_cat, tar_cat)
                )
                src_vts_dir = osp.join(
                    d_p.dt4dh_obj, Path(cross_cat_map).stem.split("_")[0], "corres"
                )  # Typo in original: 'stemspec_embed'
                src = Path(self.all_shapes[i]).stem
                tar = Path(self.all_shapes[j]).stem
                src_canonical = read_vts(src_vts_dir, src)
                tar_vts_dir = osp.join(
                    d_p.dt4dh_obj, Path(cross_cat_map).stem.split("_")[1], "corres"
                )
                tar_canonical = read_vts(tar_vts_dir, tar)
                cat_canonical = read_vts(
                    Path(cross_cat_map).parent, Path(cross_cat_map).stem
                )
                p2p_tup = list(zip(src_canonical, tar_canonical[cat_canonical]))
                p2p_tuple = np.asarray(p2p_tup)
            else:
                p2p_tuple = get_p2p_tup_from_vts(
                    osp.join(self.corr_dir), src_name, tar_name
                )
            src_evecs = numpied(self.all_ops[src_m.name_str][4][0])
            tar_evecs = numpied(self.all_ops[tar_m.name_str][4][0])
            p2p_map, fmap = refine_p2p_21_via_fmap(
                src_evecs[:, : self.n_test_ev],
                tar_evecs[:, : self.n_test_ev],
                p2p_tuple,
                return_fmap=True,
            )
            self.all_p2p_map[str(i) + "_" + str(j)] = p2p_map
            self.all_f_map[str(i) + "_" + str(j)] = fmap
            tar_mass = self.all_ops[tar_m.name_str][1][0]
            if tar_mass.squeeze().ndim == 1:
                tar_v_mass = np.diagflat(numpied(tar_mass))
            else:
                tar_v_mass = numpied(tar_mass)
            pulled_v = get_deform_spec_coeff_transf(
                src_evecs[:, : self.n_test_ev],
                tar_evecs[:, : self.n_test_ev],
                tar_v_mass,
                tar_m.verts,
                fmap=fmap,
            )
            self.all_pulled_v[str(i) + "_" + str(j)] = pulled_v
            sm_jac = src_m.frame_inv @ (
                src_m.get_extrinsic_frame(pulled_v).astype(np.float64)
            )
            self.all_sm_jac[str(i) + "_" + str(j)] = sm_jac
            sm_jac_arap = get_only_rotation_np(sm_jac)
            self.all_arap_jac_int[str(i) + "_" + str(j)] = sm_jac_arap
            spec_embed = (
                src_evecs[:, :40] @ np.linalg.pinv(src_evecs[:, :40]) @ src_m.verts
            )
            src_m.spec_embed = spec_embed


def sparse_batch_collate_disp(batches: list):
    out_keys = [
        "src_v",
        "src_f",
        "src_mass",
        "src_Lap",
        "src_evals",
        "src_evecs",
        "src_gradX",
        "src_gradY",
        "spec_embed",
        "src_f_basis",
        "arap_to_gt_jac",
        "sm_R",
        "gt_disp",
        "src_div",
        "src_lap_op",
        "name_str",
    ]
    out_dict = defaultdict(list)
    for batch in batches:
        for key, item in zip(out_keys, batch):
            if isinstance(item, coo_matrix):
                out_dict[key].append(sparse_np_to_torch(item.tocoo()))
            else:
                out_dict[key].append(item)

    for key, items in out_dict.items():
        if key != "name_str":
            out_dict[key] = torch.stack(items, dim=0)
    return out_dict


def sparse_batch_collate(batches: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    out_keys = [
        "src_v",
        "src_f",
        "src_cent",
        "src_p",
        "src_he_vec",
        "he_to_v_op",
        "src_f_basis",
        "src_fn",
        "src_div",
        "src_lap",
        "tar_mass",
        "tar_evecs",
        "frame_inv",
        "e_f_inc_ar",
        "edges",
        "tar_p",
        "tar_n",
        "tar_v",
        "gt_jac",
        "combo_idx",
    ]
    out_dict = defaultdict(list)

    for batch in batches:
        for key, item in zip(out_keys, batch):
            if isinstance(item, coo_matrix):
                if key == "he_to_v_op":
                    out_dict[key].append(sparse_np_to_tsp(item.tocoo()))
                else:
                    out_dict[key].append(sparse_np_to_torch(item.tocoo()))
            else:
                out_dict[key].append(item)

    for key, items in out_dict.items():
        if key == "combo_idx":
            continue
        elif key == "he_to_v_op":
            from torch_sparse import cat as tsp_cat

            out_dict[key] = tsp_cat(items, dim=(0, 1))
        else:
            out_dict[key] = torch.stack(items, dim=0)
    return out_dict


def dfn_sparse_batch_collate(batches: list):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    out_keys = [
        "src_v",
        "src_f",
        "tar_v",
        "tar_f",
        "tar_evecs",
        "tar_mass",
        "src_f_basis",
        "src_div",
        "src_lap_op",
        "gt_jac",
        "p2p_map",
        "sm_J",
        "sm_J_V",
        "frame_inv",
        "src_mass",
        "src_lap",
        "src_evals",
        "src_evecs",
        "src_gradX",
        "src_gradY",
        "pulled_v",
        "pair_name",
    ]
    out_dict = defaultdict(list)

    for batch in batches:
        for key, item in zip(out_keys, batch):
            out_dict[key].append(item)

    for key, items in out_dict.items():
        if key == "pair_name":
            continue
        out_dict[key] = torch.stack(items, dim=0)

    return out_dict


def get_train_test_names(data_name):
    if data_name == "faust_o" or data_name == "faust_r":
        all_names = ["tr_reg_%03d" % i for i in range(0, 100)]
        train_names = all_names[:20]
        test_names = all_names[80:]
    elif data_name == "surreal":
        data_dir = d_p.surreal_o_obj
        train_names = [Path(i).stem for i in sorted(glob(osp.join(data_dir, "*.obj")))]
        if hasattr(d_p, "surreal_val_obj"):
            test_names = [
                Path(i).stem
                for i in sorted(glob(osp.join(d_p.surreal_val_obj, "*.obj")))
            ]
        else:
            test_names = None
    elif data_name == "scape_r" or data_name == "scape_o" or data_name == "scape_ra":
        train_names = ["mesh%03d" % i for i in range(0, 51)]
        test_names = ["mesh%03d" % i for i in range(52, 72)]
    elif data_name == "shrec19_r":
        train_names = d_p.shrec19_names
        test_names = train_names
    elif "dt4dh_r" in data_name.lower():
        # name will look like dt4dh_r_cat1_cat2
        cat_1 = data_name.split("_")[2]
        cat_2 = data_name.split("_")[3]
        src_names = sorted(glob(osp.join(d_p.dt4dh_obj, f"{cat_1}", "*.obj")))
        tar_names = sorted(glob(osp.join(d_p.dt4dh_obj, f"{cat_2}", "*.obj")))
        train_names = test_names = src_names + tar_names

    return train_names, test_names


def get_dataset_dir(data_name):
    if data_name == "faust_o":
        data_dir = d_p.faust_o_obj
    elif data_name == "faust_r":
        data_dir = d_p.faust_r_obj
    elif data_name == "surreal":
        data_dir = d_p.surreal_o_obj
    elif data_name == "scape_r":
        data_dir = d_p.scape_r_obj
    elif data_name == "scape_ra":
        data_dir = d_p.scape_ra_obj
    elif data_name == "scape_o":
        data_dir = d_p.scape_o_obj
    elif data_name == "shrec19_r":
        data_dir = d_p.shrec19_r_obj
    elif "dt4dh_r" in data_name.lower():
        data_dir = d_p.dt4dh_obj
    return data_dir


def get_corr_dir(data_name):
    if data_name == "faust_r":
        corr_dir = d_p.faust_r_corr
    elif data_name == "scape_r" or data_name == "scape_ra":
        corr_dir = d_p.scape_r_corr
    elif data_name == "shrec19_r":
        corr_dir = d_p.shrec19_r_corr
    elif "dt4dh_r" in data_name.lower():
        corr_dir = d_p.dt4d_r_corr
    return corr_dir


def get_dataset_class(data_name, **kwargs):
    """
    Factory function to get the appropriate dataset and dataloader.

    Args:
        data_name (str): Name of the dataset.
        **kwargs:
            enc_name (str): Encoder name ('point' or 'dfn').
            batch_size (int): Batch size.
            spec_inp (bool): Whether to use spectral input.
            mode (str): 'train' or 'test'.
            collate_fn (callable): Custom collate function.

    Returns:
        tuple: A tuple containing the dataset class instance and the dataloader.
    """
    enc_name = kwargs.get("enc_name", "dfn")
    batch_size = kwargs["batch_size"]
    spec_inp = kwargs["spec_inp"]
    mode = kwargs["mode"]
    my_collate_fn = kwargs.get("collate_fn", dfn_sparse_batch_collate)
    train_names, test_names = get_train_test_names(data_name)
    data_dir = get_dataset_dir(data_name)
    if enc_name.lower() == "point":  # Legacy, not used in the paper
        if mode == "train":
            dset_cls = BatchJacDataset(
                data_dir, train_names, is_permut=True, spec_inp=spec_inp
            )
            dset_dl = torch.utils.data.DataLoader(
                dset_cls,
                batch_size=batch_size,
                collate_fn=sparse_batch_collate,
                shuffle=True,
                num_workers=batch_size,
                persistent_workers=True,
                pin_memory=False,
                drop_last=True,
            )
        else:
            dset_cls = BatchJacDataset(
                data_dir, test_names, is_permut=True, spec_inp=spec_inp
            )
            dset_dl = torch.utils.data.DataLoader(
                dset_cls, batch_size=1, collate_fn=sparse_batch_collate
            )
    elif enc_name.lower() == "dfn":
        if "_r" in data_name:
            corr_dir = get_corr_dir(data_name)
            if (
                mode == "train"
            ):  # Remeshed datasets are only for testing/evaluation in the paper
                dset_cls = RemeshedDfnDataset(
                    data_name,
                    data_dir,
                    train_names,
                    corr_dir=corr_dir,
                    is_permut=True,
                    spec_inp=spec_inp,
                )
            else:
                if "dt4dh" in data_name:
                    all_shapes = train_names
                    dset_cls = RemeshedDfnDataset(
                        data_name,
                        data_dir=data_dir,
                        all_shapes=all_shapes,
                        corr_dir=corr_dir,
                        is_permut=False,
                        spec_inp=spec_inp,
                    )
                else:
                    dset_cls = RemeshedDfnDataset(
                        data_name,
                        data_dir,
                        test_names,
                        corr_dir=corr_dir,
                        is_permut=False,
                        spec_inp=spec_inp,
                    )

        else:
            if mode == "train":
                dset_cls = DfnBatchJacDataset(
                    data_dir,
                    train_names,
                    is_permut=True,
                    spec_inp=spec_inp,
                    varying_fm=False,
                )
            else:
                dset_cls = DfnBatchJacDataset(
                    data_dir,
                    test_names,
                    is_permut=True,
                    spec_inp=spec_inp,
                    varying_fm=False,
                )

        if mode == "train":
            dset_dl = torch.utils.data.DataLoader(
                dset_cls,
                batch_size=batch_size,
                collate_fn=my_collate_fn,
                shuffle=True,
                num_workers=16,
                persistent_workers=True,
                pin_memory=False,
                drop_last=True,
            )
        else:
            dset_dl = torch.utils.data.DataLoader(
                dset_cls, batch_size=1, collate_fn=my_collate_fn
            )
    return dset_cls, dset_dl
