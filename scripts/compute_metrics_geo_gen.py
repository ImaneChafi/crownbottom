import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# -------------------------
# Distance functions
# -------------------------

def chamfer_distance(mesh_a, mesh_b):
    A = mesh_a.vertices
    B = mesh_b.vertices

    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    dists_A_to_B, _ = tree_B.query(A)
    dists_B_to_A, _ = tree_A.query(B)

    return np.mean(dists_A_to_B) + np.mean(dists_B_to_A), dists_A_to_B, dists_B_to_A


def hausdorff_distance(mesh_a, mesh_b):
    A = mesh_a.vertices
    B = mesh_b.vertices

    tree_A = cKDTree(A)
    tree_B = cKDTree(B)

    dists_A_to_B, _ = tree_B.query(A)
    dists_B_to_A, _ = tree_A.query(B)

    return max(np.max(dists_A_to_B), np.max(dists_B_to_A)), dists_A_to_B, dists_B_to_A


# -------------------------
# Utility
# -------------------------

def extract_id(filename):
    return filename[:7]  # e.g. "6087-43"


# -------------------------
# Main evaluation
# -------------------------

def evaluate_folders(gen_dir, gt_dir):
    gen_files = {extract_id(f): f for f in os.listdir(gen_dir) if f.endswith(".stl")}
    gt_files  = {extract_id(f): f for f in os.listdir(gt_dir)  if f.endswith(".stl")}

    common_ids = sorted(set(gen_files.keys()) & set(gt_files.keys()))

    chamfers = []
    hausdorffs = []

    all_chamfer_points = []
    all_hausdorff_points = []

    for sid in common_ids:
        gen_mesh = trimesh.load(os.path.join(gen_dir, gen_files[sid]), process=False)
        gt_mesh  = trimesh.load(os.path.join(gt_dir,  gt_files[sid]),  process=False)

        cd, dA, dB = chamfer_distance(gen_mesh, gt_mesh)
        hd, _, _ = hausdorff_distance(gen_mesh, gt_mesh)

        chamfers.append(cd)
        hausdorffs.append(hd)

        all_chamfer_points.extend(dA)
        all_chamfer_points.extend(dB)
        all_hausdorff_points.extend(dA)
        all_hausdorff_points.extend(dB)

        print(f"{sid} | Chamfer: {cd:.6f} | Hausdorff: {hd:.6f}")

    chamfers = np.array(chamfers)
    hausdorffs = np.array(hausdorffs)

    # -------------------------
    # Aggregated metrics
    # -------------------------

    mean_chamfer = np.mean(chamfers)
    std_chamfer  = np.std(chamfers)

    mean_hausdorff = np.mean(hausdorffs)
    std_hausdorff  = np.std(hausdorffs)

    print("\n=== FINAL RESULTS ===")
    print(f"Chamfer: {mean_chamfer:.6f} ± {std_chamfer:.6f}")
    print(f"Hausdorff: {mean_hausdorff:.6f} ± {std_hausdorff:.6f}")

    # -------------------------
    # Histograms
    # -------------------------

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.hist(all_chamfer_points, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Chamfer Distance Distribution\nmean: {mean_chamfer:.4f} ± {std_chamfer:.4f}")
    plt.xlabel("Chamfer distance")
    plt.ylabel("Frequency")

    plt.subplot(1,2,2)
    plt.hist(all_hausdorff_points, bins=30, color='salmon', edgecolor='black')
    plt.title(f"Hausdorff Distance Distribution\nmean: {mean_hausdorff:.4f} ± {std_hausdorff:.4f}")
    plt.xlabel("Hausdorff distance")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    gen_folder = "/Users/imanechafi/Desktop/PhD/Journal/journal_2026/premolar/ori_data/geometric_gen/total"
    gt_folder  = "/Users/imanechafi/Desktop/PhD/Journal/journal_2026/premolar/ori_data/gt/total"

    evaluate_folders(gen_folder, gt_folder)
