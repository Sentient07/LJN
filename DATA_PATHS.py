# DATA_PATHS.py
from itertools import combinations, permutations
import os
from pathlib import Path
from glob import glob
import os.path as osp

faust_r_obj = "/home/ramana/Datasets/faust_remeshed/obj"
faust_r_corr = "/home/ramana/Datasets/faust_remeshed/corres"
faust_r_geod = "/home/ramana/Datasets/faust_remeshed/geod"
faust_pairs = [
    ("tr_reg_%03d" % i, "tr_reg_%03d" % j) for (i, j) in combinations(range(80, 100), 2)
]
faust_pairs_full = [
    ("tr_reg_%03d" % i, "tr_reg_%03d" % j) for (i, j) in permutations(range(80, 100), 2)
]
faust_o_obj = "/mnt/disk2/ramana/data/MPI-FAUST/training/registrations/obj"
faust_eval_n = [("tr_reg_%03d" % i) for i in range(80, 100)]

scape_r_obj = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SCAPE_r/shapes_obj/"
)
scape_ra_obj = "/mnt/disk2/ramana/data/scape_remesh/ply_al2_obj"
scape_r_corr = "/mnt/disk2/ramana/data/scape_remesh/corres/"
scape_r_geod = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SCAPE_r/test_geod/"
)
scape_pairs = [
    ("mesh%03d" % i, "mesh%03d" % j) for (i, j) in combinations(range(52, 72), 2)
]
scape_pairs_full = [
    ("mesh%03d" % i, "mesh%03d" % j) for (i, j) in permutations(range(52, 72), 2)
]
scape_o_obj = "/mnt/disk2/ramana/data/SCAPE_ORIGINAL/obj_rot/"
dt4dh_obj = "/mnt/disk2/ramana/data/DT4D_Match/DeformingThings4DMatching"
dt4dh_r_corr = (
    "/mnt/disk2/ramana/data/DT4D_Match/DeformingThings4DMatching/cross_category_corres"
)
dt4dh_r_obj = "/mnt/disk2/ramana/data/DT4D_Match/DeformingThings4DMatching/Eval/Meshes"
dt4dh_r_niso_pairs = [
    i.rstrip().split("_")
    for i in open(
        "/mnt/disk2/ramana/data/DT4D_Match/DeformingThings4DMatching/eval_list_noniso.txt"
    ).readlines()
]

smal_r_tr_obj = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SMAL_r/shapes/train"
)
smal_r_corr = "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SMAL_r/corres"
smal_r_geod = "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SMAL_r/geod"

smal_r_te_obj = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SMAL_r/shapes/test_scaled/"
)
all_files_smal = sorted(glob(osp.join(smal_r_te_obj, "*.obj")))
smal_r_pairs = [
    (Path(all_files_smal[i]).stem, Path(all_files_smal[j]).stem)
    for i, j in combinations(range(len(all_files_smal)), 2)
]
smal_r_pairs_full = [
    (Path(all_files_smal[i]).stem, Path(all_files_smal[j]).stem)
    for i, j in permutations(range(len(all_files_smal)), 2)
]

surreal_o_obj = "/mnt/disk2/ramana/data/SURREAL/random_subset_surreal_2k/obj"
surreal_train_txt = (
    "/mnt/disk2/ramana/data/SURREAL/random_subset_surreal_2k/surreal2k_train.txt"
)

# shrec19_pair_f = '/mnt/disk2/ramana/data/SHREC19/eval_pairs.txt'


sumner_cat_obj = "/mnt/disk2/ramana/data/cat-poses"
mano_obj = "/mnt/disk2/ramana/data/hands_registration_closed"


shrec19_pair_f = [
    i.rstrip().split(",")
    for i in open(
        "/mnt/disk2/ramana/data/SHREC19/Remeshed/eval_pair_final.txt"
    ).readlines()
]
shrec19_names = [i for i in range(1, 45)]
shrec19_o_obj = "/mnt/disk2/ramana/data/SHREC19/ply_rot_obj/"
shrec19_r_obj = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SHREC_r/shapes_obj"
)
shrec19_r_geod = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SHREC_r/test_geod"
)
shrec19_r_corr = (
    "/home/ramana/NewProject/Baselines/AttentiveFMaps/data/SHREC_r/correspondences"
)

shrec19_r_pairs = []
for i in os.listdir(shrec19_r_corr):
    p_str = Path(i).stem.split("_")
    src_, tar_ = int(p_str[0]), int(p_str[1])
    if src_ != 40 and tar_ != 40:
        shrec19_r_pairs.append((p_str[0], p_str[1]))

dt4d_ulrssm_maps = (
    "/home/ramana/NewProject/Baselines/ULRSSM/results/dt4d_inter_class/visualization"
)
dt4d_pairs_all = [
    i.split("-") for i in os.listdir(dt4d_ulrssm_maps) if i.endswith(".mat")
]
dt4d_pairs_all = [(Path(i[0]).stem, Path(i[1]).stem) for i in dt4d_pairs_all]

dt4dh_eval_obj = (
    "/mnt/disk2/ramana/data/DT4D_Match/DeformingThings4DMatching/Eval/AllMeshes/"
)
