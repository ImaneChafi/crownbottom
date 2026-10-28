import os
import h5py
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt

# ============================================================
# Utilities
# ============================================================

def normalize_point_cloud(pc):
    pc = pc - np.mean(pc, axis=0)
    scale = np.max(np.linalg.norm(pc, axis=1))
    pc = pc / scale
    return pc

def resample_point_cloud(pc, n_points=2048):
    if pc.shape[0] > n_points:
        idx = np.arange(n_points)  # deterministic: take first N points
        pc = pc[idx]
    elif pc.shape[0] < n_points:
        idx = np.random.choice(pc.shape[0], n_points, replace=True)
        pc = pc[idx]
    return pc

# ============================================================
# Loading
# ============================================================

def load_reference_from_h5(h5_path, dataset_name, n_points=2048):
    with h5py.File(h5_path, "r") as f:
        pcs = f[dataset_name][:]
    if pcs.ndim == 2:
        pcs = pcs[None, ...]
    pcs_list = []
    for pc in pcs:
        pc = resample_point_cloud(pc, n_points)
        pc = normalize_point_cloud(pc)
        pcs_list.append(pc.astype(np.float64))
    pcs = np.stack(pcs_list, axis=0)
    return pcs

def load_generated_from_dir(directory, n_points=2048):
    pcs = []
    filenames = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".ply"):
            path = os.path.join(directory, fname)
            mesh = trimesh.load(path, process=False)
            pts = np.asarray(mesh.vertices)
            pts = resample_point_cloud(pts, n_points)
            pts = normalize_point_cloud(pts)
            pcs.append(pts.astype(np.float64))
            filenames.append(fname)
    pcs = np.stack(pcs, axis=0)
    return pcs, filenames

# ============================================================
# ICP alignment
# ============================================================

def align_point_clouds(reference_pc, sample_pc, threshold=0.1):
    ref_o3d = o3d.geometry.PointCloud()
    ref_o3d.points = o3d.utility.Vector3dVector(reference_pc)
    samp_o3d = o3d.geometry.PointCloud()
    samp_o3d.points = o3d.utility.Vector3dVector(sample_pc)
    
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        samp_o3d, ref_o3d, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    aligned = np.asarray(samp_o3d.transform(reg_p2p.transformation).points)
    return aligned

# ============================================================
# Metrics
# ============================================================

def chamfer_distance(a, b):
    nn_a = NearestNeighbors(n_neighbors=1).fit(b)
    nn_b = NearestNeighbors(n_neighbors=1).fit(a)
    d1, _ = nn_a.kneighbors(a)
    d2, _ = nn_b.kneighbors(b)
    return np.mean(d1**2) + np.mean(d2**2)

def minimum_matching_distance(sample_pcs, ref_pcs):
    matched_dists = []
    for ref in ref_pcs:
        dists = [chamfer_distance(sample, ref) for sample in sample_pcs]
        matched_dists.append(min(dists))
    return np.mean(matched_dists), np.array(matched_dists)

def coverage(sample_pcs, ref_pcs):
    matched_refs = set()
    for sample in sample_pcs:
        dists = [chamfer_distance(sample, ref) for ref in ref_pcs]
        matched_refs.add(int(np.argmin(dists)))
    return len(matched_refs) / len(ref_pcs)

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    def occupancy_grid(pcs):
        grid = np.zeros((resolution,) * 3)
        for pc in pcs:
            idx = ((pc + 0.5) * resolution).astype(int)
            idx = np.clip(idx, 0, resolution - 1)
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] += 1
        grid /= np.sum(grid)
        return grid.flatten()
    P = occupancy_grid(sample_pcs)
    Q = occupancy_grid(ref_pcs)
    M = 0.5 * (P + Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))

# ============================================================
# Additional Metrics (All-pair Chamfer & Hausdorff)
# ============================================================

def hausdorff_distance(a, b):
    d_ab = directed_hausdorff(a, b)[0]
    d_ba = directed_hausdorff(b, a)[0]
    return max(d_ab, d_ba)

def chamfer_distance_original(a, b):
    nn_a = NearestNeighbors(n_neighbors=1).fit(b)
    nn_b = NearestNeighbors(n_neighbors=1).fit(a)
    d1, _ = nn_a.kneighbors(a)
    d2, _ = nn_b.kneighbors(b)
    return np.mean(d1**2) + np.mean(d2**2)

def compute_all_pair_metrics(sample_pcs, ref_pcs):
    chamfer_list = []
    hausdorff_list = []

    for sample in sample_pcs:
        for ref in ref_pcs:
            chamfer_list.append(chamfer_distance_original(sample, ref))
            hausdorff_list.append(hausdorff_distance(sample, ref))

    cham_mean = np.mean(chamfer_list)
    cham_std = np.std(chamfer_list)
    haus_mean = np.mean(hausdorff_list)
    haus_std = np.std(hausdorff_list)

    return (cham_mean, cham_std), (haus_mean, haus_std), chamfer_list, hausdorff_list

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    REF_H5 = "/Users/imanechafi/Desktop/PhD/Journal/journal_2026/canine/gen_data/canine.h5"
    DATASET_NAME = "poisson_2048"
    GEN_DIR = "/Users/imanechafi/Desktop/PhD/Journal/journal_2026/canine/results/3d"
    N_POINTS = 2048
    OUTPUT_ALIGNED_DIR = "/Users/imanechafi/Desktop/PhD/Journal/journal_2026/canine/gen_data/aligned_dir"
    os.makedirs(OUTPUT_ALIGNED_DIR, exist_ok=True)

    # Load
    ref_pcs = load_reference_from_h5(REF_H5, DATASET_NAME, n_points=N_POINTS)
    sample_pcs, filenames = load_generated_from_dir(GEN_DIR, n_points=N_POINTS)

    # Align and save
    aligned_sample_pcs = []
    for i, sample in enumerate(sample_pcs):
        aligned = align_point_clouds(ref_pcs[0], sample)
        aligned_sample_pcs.append(aligned)
        save_path = os.path.join(OUTPUT_ALIGNED_DIR, f"aligned_generated_{filenames[i]}")
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(aligned.astype(np.float64))
        o3d.io.write_point_cloud(save_path, pc_o3d)
    
    aligned_sample_pcs = np.stack(aligned_sample_pcs, axis=0)

    # Save aligned reference clouds
    for i, ref in enumerate(ref_pcs):
        save_path = os.path.join(OUTPUT_ALIGNED_DIR, f"aligned_reference_{i}.ply")
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(ref.astype(np.float64))
        o3d.io.write_point_cloud(save_path, pc_o3d)

    print(f"All aligned reference and generated PLY files saved to {OUTPUT_ALIGNED_DIR}")

    # -----------------------
    # Compute metrics
    # -----------------------
    print("\nComputing MMD...")
    mmd, _ = minimum_matching_distance(aligned_sample_pcs, ref_pcs)
    print("MMD:", mmd)

    print("\nComputing Coverage...")
    cov = coverage(aligned_sample_pcs, ref_pcs)
    print("COV:", cov)

    print("\nComputing JSD...")
    jsd = jsd_between_point_cloud_sets(aligned_sample_pcs, ref_pcs)
    print("JSD:", jsd)

    print("\nComputing Average Hausdorff and Chamfer distances (all-pairs)...")
    (cham_mean, cham_std), (haus_mean, haus_std), chamfer_list, hausdorff_list = compute_all_pair_metrics(
        aligned_sample_pcs, ref_pcs
    )

    print(f"Average Chamfer distance: {cham_mean:.6f} ± {cham_std:.6f}")
    print(f"Average Hausdorff distance: {haus_mean:.6f} ± {haus_std:.6f}")

    # -----------------------
    # Visualization
    # -----------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(chamfer_list, bins=20, color='skyblue', edgecolor='k')
    plt.title("Chamfer Distance Distribution")
    plt.xlabel("Chamfer distance")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    plt.hist(hausdorff_list, bins=20, color='salmon', edgecolor='k')
    plt.title("Hausdorff Distance Distribution")
    plt.xlabel("Hausdorff distance")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
