# implement crown bottom generation with geometrical method
# the script generates a crown bottom with a die mesh and its marginline, both registered
# with lowerjaw registration matrix

# step1: cut input die mesh with input marginline
# step2: blockout undercut from marginline undercut_start
# step3: add cement gap and smooth cut die for the points distant from marginline cement_start
# step4: generate offset boundary band
# step5: merge both meshes obtained from step 3 and 4

import pyvista as pv
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import os, fnmatch
import vtk
from scipy.spatial import cKDTree
import pymeshlab
from scipy import interpolate
import open3d as o3d
import shutil
from time import time
import math

def get_boundary_ids(mesh):
    idFilter = vtk.vtkIdFilter()
    idFilter.SetInputData(mesh)
    idFilter.SetPointIdsArrayName("ids")
    idFilter.SetPointIds(True)
    idFilter.SetCellIds(False)
    idFilter.Update()

    edges = vtk.vtkFeatureEdges()
    edges.SetInputConnection(idFilter.GetOutputPort())
    edges.BoundaryEdgesOn()
    edges.ManifoldEdgesOff()
    edges.NonManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()

    array = edges.GetOutput().GetPointData().GetArray("ids")
    n = edges.GetOutput().GetNumberOfPoints()
    boundaryIds = []
    for i in range(n):
        boundaryIds.append(array.GetValue(i))
    return boundaryIds

def getNewPoints(arrayOfPoints):
    listNewPoints = []
    nb_points = arrayOfPoints.shape[0]
    traversal, distances = trimesh.points.tsp(arrayOfPoints, start=0)
    min_dist = min(distances)
    for i in range(nb_points):
        newPoint = None
        if (i == nb_points - 1):
            dist_end = math.dist(arrayOfPoints[traversal[0]], arrayOfPoints[traversal[-1]])
            if dist_end > min_dist:
                newPoint = 0.5*(arrayOfPoints[traversal[-1]] + arrayOfPoints[traversal[0]])
        else:
            newPoint = 0.5*(arrayOfPoints[traversal[i]] + arrayOfPoints[traversal[i + 1]])

        if newPoint is not None:
            listNewPoints.append(newPoint)

    ndArrayNewPoints = np.array(listNewPoints)
    concatenatedNewArray = np.concatenate((ndArrayNewPoints, arrayOfPoints))

    return concatenatedNewArray

def cutting_die(die_path, margin_path, die_type, extracted, thickness=0.05):
    # suppose both die mesh and marginline have been registered with lowerjaw registration matrix
    die_folder = os.path.dirname(die_path)
    die_file = os.path.basename(die_path)
    die_mesh = pv.read(die_path)
    die_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    mean_die_normal = die_mesh['Normals'].mean(axis=0)
    margin_mesh = pv.read(margin_path)
    margin_points = margin_mesh.points
    print('initial marginline size = ', margin_points.shape[0])
    # refine marginline if necessary
    sampling_size = 500
    while margin_points.shape[0] < sampling_size:
        # double size is really necessary?
        margin_points = getNewPoints(margin_points)

    print('after refining, marginline size =  ', margin_points.shape[0])

    margin_size= margin_points.shape[0]
    sampled_points = margin_points
    margin_points_index = range(margin_size)
    if margin_size > sampling_size:
        # farthest point sampling
        pcd_marginline = o3d.geometry.PointCloud()
        pcd_marginline.points = o3d.utility.Vector3dVector(margin_points)
        sampled_pcd = pcd_marginline.farthest_point_down_sample(sampling_size)
        sampled_points = np.asarray(sampled_pcd.points)

    # resorted sampled_points
    # reorder sampled_points in different way depending on extracted
    margin_center = sampled_points.mean(axis=0)
    origine = margin_center
    if extracted:
        traversal, distances = trimesh.points.tsp(sampled_points, start=0)
        ordered_sampled_points = sampled_points[traversal]
    else:
        normal = [0., 0., 1.0]
        if 'upper' in die_type.lower():
            normal = [0., 0., -1.0]
        ordered_sampled_points = trimesh.points.radial_sort(sampled_points, origine, normal)

    sampled_points = ordered_sampled_points
    # find boundary length
    boundary_line = pv.lines_from_points(sampled_points, close=True)
    boundary_lengths = boundary_line.compute_cell_sizes()
    boundary_length = sum(boundary_lengths['Length'])
    boundary_line.save(os.path.join(die_folder, 'sampled_marginline.ply'))
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(die_mesh.points)
    _, indices = nbrs.kneighbors(sampled_points)
    # create an offset surface mesh
    points_margin = []
    points_internal = []
    points_external = []

    epsilon_norm = 1.e-5
    for i, index in enumerate(indices):
        closest_margin_point_index = index[0]
        normal = die_mesh['Normals'][closest_margin_point_index]
        norm = np.linalg.norm(normal)
        if norm < epsilon_norm:
            #print('warning: very small normal found! norm = ', norm)
            normal = normal / epsilon_norm
        else:
            normal = normal / norm
        closest_margin_point = sampled_points[i]
        points_margin.append(closest_margin_point)
        internal_point = closest_margin_point - 0.5*thickness*normal
        external_point = closest_margin_point + 0.5*thickness*normal
        dist_int = math.dist(internal_point[0:2], margin_center[0:2])
        dist_ext = math.dist(external_point[0:2], margin_center[0:2])
        if dist_ext < dist_int:
            print('one point with inverse normal!')
            internal_point[0] = closest_margin_point[0]
            internal_point[1] = closest_margin_point[1]
            external_point[0] = closest_margin_point[0]
            external_point[1] = closest_margin_point[1]
        points_internal.append(internal_point)
        points_external.append(external_point)

    # we need to smooth offset points
    # close margin
    points_margin.append(points_margin[0])
    points_internal.append(points_internal[0])
    points_external.append(points_external[0])

    # using 3 polygons to create a surface mesh
    total_points = np.vstack((points_internal, points_margin, points_external))
    # create a grid for clipping
    grid_clipping = pv.StructuredGrid()
    grid_clipping.points = total_points
    grid_clipping.dimensions = len(points_margin), 3, 1
    grid_clipping = grid_clipping.cast_to_unstructured_grid()
    grid_clipping = grid_clipping.extract_surface()
    grid_clipping.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    mean_grid_normal = grid_clipping['Normals'].mean(axis=0)
    grid_clipping.save(os.path.join(die_folder, 'cut_grid.stl'))

    invert = True
    if mean_die_normal[2]*mean_grid_normal[2] > 0:
        invert = False
    die_mesh_top = die_mesh.clip_surface(grid_clipping, invert=invert)
    die_mesh_top = die_mesh_top.extract_largest()

    boundary_index = get_boundary_ids(die_mesh_top)
    bd_pts = die_mesh_top.points[boundary_index]
    print('boundary size = ', len(boundary_index))
    bd_mesh = die_mesh_top.extract_points(boundary_index)
    bd_mesh = bd_mesh.extract_surface()

    # detect if holes exists
    largest_bd = bd_mesh.extract_largest()
    largest_bd = largest_bd.extract_surface()
    largest_ncells = largest_bd.n_cells
    bd_ncells = bd_mesh.n_cells
    if largest_ncells < bd_ncells:
        print('largest boundary cell size = ', largest_ncells)
        print('cut die boundary cell size: ', bd_ncells)
        print('Error:  there are holes in the cut die mesh!')
        die_folder = os.path.dirname(die_path)
        largest_bd.save(os.path.join(die_folder, 'largest_boundary.stl'))
        bd_mesh.save(os.path.join(die_folder, 'boundary.stl'))
        die_mesh_top.save(os.path.join(die_folder, 'debug_cut_die.stl'))
        print("Error: crown bottom generation fails in cutting die!")

    edge_length = 0.075
    target_bd_size = int(boundary_length/edge_length)
    print('target_bd_size = ', target_bd_size)

    if target_bd_size < min(len(boundary_index), margin_size):
        # sample target_bd_size points from bd_pts, should be follow a weights
        pcd_margin = o3d.geometry.PointCloud()
        pcd_margin.points = o3d.utility.Vector3dVector(margin_points)
        fps_sampled_points = pcd_margin
        if margin_size > target_bd_size:
            fps_sampled_points = pcd_margin.farthest_point_down_sample(num_samples=target_bd_size)

        sampled_bd = fps_sampled_points
        kmeans_bd = KMeans(n_clusters=target_bd_size, n_init=1, init=sampled_bd.points).fit(bd_pts)
        labels = np.unique(kmeans_bd.labels_)
        clustered_cb_bd = []
        linked_points = []
        linked_keys = []
        if len(labels) < target_bd_size:
            target_bd_size = len(labels)

        for i in range(target_bd_size):
            index = kmeans_bd.labels_ == labels[i]
            # find the closest point to the polygone_cb_bottom.points[i]
            dists = []
            idx = np.where(index == True)[0]
            linked_points.append(idx)
            for ix in idx:
                dists.append(math.dist(bd_pts[ix], sampled_bd.points[i]))
            index_selected = idx[np.argmin(dists)]
            linked_keys.append(index_selected)
            clustered_cb_bd.append(bd_pts[index_selected])

        for iseg in range(target_bd_size):
            points_sorted = bd_pts[linked_points[iseg]]
            proxy = bd_pts[linked_keys[iseg]]
            # tie points_sorted to bd_pts
            index_mapped = get_index_mapped(bd_pts, points_sorted, tol=0.001)
            #print('iseg = ', iseg, '  index_mapped size = ', len(index_mapped))
            bd_pts[index_mapped] = proxy

    die_mesh_top.points[boundary_index] = bd_pts

    return die_mesh_top, invert

def get_index_mapped(dest_centers, ref_centers, tol=5):  # using open3d
    pcd_dest = o3d.geometry.PointCloud()
    pcd_dest.points = o3d.utility.Vector3dVector(dest_centers)
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_centers)
    dist = pcd_dest.compute_point_cloud_distance(pcd_ref)
    shortest_d = np.asarray(dist)
    # compute the shortest distance
    shortest_index = sorted(range(len(shortest_d)), key=lambda k: shortest_d[k])
    shortest_d.sort()
    if len(np.where(shortest_d < tol)[0]) > 0:
        idx_max = np.max(np.where(shortest_d < tol), axis=1)
        return shortest_index[0:idx_max[0]+1]
    else:
        return []

def add_cement_gap(cut_die, cement_start, cement_gap):
    boundary_pts_id = get_boundary_ids(cut_die)
    # dealing with cement gap
    boundary_points = cut_die.points[boundary_pts_id]
    index_cement = get_index_mapped(cut_die.points, boundary_points, tol=cement_start)
    mask = np.zeros(len(cut_die.points), dtype=bool)
    for idx in index_cement:
        mask[idx] = True

    die_cut_cemented, _idx = cut_die.remove_points(mask)
    # do an offset before as cement gap
    die_cut_cemented.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    bd_index = get_boundary_ids(die_cut_cemented)
    for index in range(die_cut_cemented.points.shape[0]):
        if index not in bd_index:
            point = die_cut_cemented.points[index]
            die_cut_cemented.points[index] = point + die_cut_cemented['Normals'][index] * cement_gap

    # take account the boundary points to get remain part
    die_tree = cKDTree(cut_die.points)
    distances, index_cement_bd = die_tree.query(die_cut_cemented.points[bd_index])
    kept_mask = np.ones(len(cut_die.points), dtype=bool)
    for idx in index_cement:
        kept_mask[idx] = False
    for idx in index_cement_bd:
        kept_mask[idx] = False

    kept_die, _idx = cut_die.remove_points(kept_mask)
    # merge cemented and kept part to get crown bottom
    final_cb = die_cut_cemented.merge(kept_die)
    # postprocessing
    final_cb = final_cb.fill_holes(2)  # there are some holes in the kept_die
    return final_cb

def blockout_undercut(cut_die, blockout_start, cement_start):
    cut_die.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    boundary_pts_id = get_boundary_ids(cut_die)
    # dealing with cement gap
    boundary_points = cut_die.points[boundary_pts_id]
    index_blockout = get_index_mapped(cut_die.points, boundary_points, tol=blockout_start)
    mask = np.zeros(len(cut_die.points), dtype=bool)
    for idx in index_blockout:
        mask[idx] = True

    die_cut_blockout, _idx = cut_die.remove_points(mask)

    uncemented_index = get_index_mapped(cut_die.points, boundary_points, tol=cement_start)

    # do an offset for each point
    #die_cut_blockout.compute_normals(cell_normals=False, point_normals=True, inplace=True)
    bd_index = get_boundary_ids(die_cut_blockout)

    for index in range(cut_die.points.shape[0]):
        normal = cut_die['Normals'][index]
        if index not in index_blockout:
            if index in uncemented_index:
                cut_die.points[index] = cut_die.points[index] + normal * 0.035
            else:
                cut_die.points[index] = cut_die.points[index] + normal * 0.015

    # postprocessing
    final_cb = cut_die.fill_holes(2)  # there are some holes in the kept_die
    return final_cb


def cement_and_smooth_cut_die(cut_die_path, smoothed_path, cement_start, cement_gap, blockout_distance, n_iter):
    cut_die = pv.read(cut_die_path)
    cemented_cb = add_cement_gap(cut_die, cement_start, cement_gap)
    #blockout_thickness = 0.025   # to compensate smoothing effect
    #blocked_cb = add_cement_gap(cemented_cb, blockout_distance, blockout_thickness)
    boundary_index = get_boundary_ids(cemented_cb)
    boundary_points = cemented_cb.points[boundary_index]
    block_index = get_index_mapped(cemented_cb.points, boundary_points, blockout_distance)
    intact_points = cemented_cb.points[block_index]
    # smoothing
    smoothed_cb = cemented_cb.smooth_taubin(n_iter, pass_band=0.05)
    # restore blocked part
    for i, idx in enumerate(block_index):
        smoothed_cb.points[idx] = intact_points[i]

    save_path = os.path.dirname(cut_die_path)
    final_cb = smoothed_cb.fill_holes(2)  # there are some holes in the kept_die
    final_cb.save(os.path.join(save_path, 'temp_cb.stl'))

    # remove duplicated faces and close vertices with pymeshlab filter
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(os.path.join(save_path, 'temp_cb.stl'))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_re_orient_faces_coherently()   # try to fix fr case 6116-22
    ms.save_current_mesh(smoothed_path)
    # clean temporary file
    os.remove(os.path.join(save_path, 'temp_cb.stl'))

def smooth_polygon_with_spline(offset_points, smooth=0.1, niter=2):
    # approximate ordered points with spline and output a set of points with same t
    for iter in range(niter):
        # smoothing offset_points using spline
        size = offset_points.shape[0]
        sqrt_size = np.sqrt(2 * size)
        tck, u = interpolate.splprep([offset_points[:, 0], offset_points[:, 1], offset_points[:, 2]], s=smooth, per=True)
        # sample points from spline
        x_, y_, z_ = interpolate.splev(u, tck)
        offset_points = np.concatenate((x_.reshape(-1, 1), y_.reshape(-1, 1), z_.reshape(-1, 1)), axis=1)

    return offset_points

def generate_extension_band(die_path, smoothed_path, die_type, extracted, offset):
    cb_without_band = pv.read(smoothed_path)
    bd_ids = get_boundary_ids(cb_without_band)
    boundary_pts = cb_without_band.points[bd_ids]
    bd_center = boundary_pts.mean(axis=0)
    die_mesh = pv.read(die_path)
    die_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    cb_without_band.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    # ordered boundary
    margin_center = boundary_pts.mean(axis=0)
    origine = margin_center
    ordered_bd_pts = boundary_pts

    if extracted:
        traversal, distances = trimesh.points.tsp(boundary_pts, start=0)
        ordered_bd_pts = boundary_pts[traversal]
    else:
        normal_ordered = [0., 0., 1.0]
        if 'upper' in die_type.lower():
            normal_ordered = [0., 0., -1.0]
        ordered_bd_pts = trimesh.points.radial_sort(boundary_pts, origine, normal_ordered)

    boundary_pts = ordered_bd_pts
    # find closest cell on the input die mesh, very sensitive to the cell normal
    # using averaged normal of all connecting cells will be more robust
    # but above method is not adequate for extracted prep
    index_cells = die_mesh.find_closest_cell(boundary_pts)
    cell_normals = die_mesh['Normals'][index_cells]

    if extracted:
        for i, pt in enumerate(boundary_pts):
            containing_cells = cb_without_band.find_containing_cell(pt)
            normals = cb_without_band['Normals'][containing_cells]
            cell_normals[i] = normals.mean(axis=0)
            cell_normals[i] = cell_normals[i] / np.linalg.norm(cell_normals[i])
            u = pt - margin_center
            norm_u = np.linalg.norm(u)
            u = u / norm_u
            if i > 0 and i < boundary_pts.shape[0] - 1:
                cell_normals[i] = 0.5*(u+cell_normals[i-1])
            elif i == boundary_pts.shape[0] - 1:    # avoid intersection
                cell_normals[i] = cell_normals[0]
            else:
                cell_normals[i] = u

    offset_points = []
    epsilon_norm = 1.e-4
    flipped = []

    for i in range(boundary_pts.shape[0]):
        point = boundary_pts[i]
        normal = cell_normals[i]
        # make horizontal border
        normal[2] = 0
        norm = np.linalg.norm(normal)
        if norm < epsilon_norm:
            normal = normal / epsilon_norm
        else:
            normal = normal / norm
        result = point + normal * offset

        offset_points.append(result)

    # create an extension with two lines
    # smoothing offset_points using spline, but too much smoothing
    offset_points = np.array(offset_points)

    # reorder offset points to avoid intersections
    #traversal, distances = trimesh.points.tsp(offset_points, start=0)
    #ordered_offset_points = offset_points[traversal]
    #offset_points = ordered_offset_points

    iters = 2
    offset_points = smooth_polygon_with_spline(offset_points, smooth=0.1, niter=iters)

    # reorder offset points to avoid intersections
    traversal, distances = trimesh.points.tsp(offset_points, start=0)
    ordered_offset_points = offset_points[traversal]
    offset_points = ordered_offset_points
    # close
    # check if start and end points are too close each other
    margin_start = boundary_pts[0]
    margin_end = boundary_pts[-1]
    dist_margin = math.dist(margin_start, margin_end)
    print('dist between two end margin: ', dist_margin)
    offset_start = offset_points[0]
    offset_end = offset_points[-1]
    dist_offset = math.dist(offset_start, offset_end)
    print('dist between two end offset: ', dist_offset)

    # switch first and second element for case 6116-22
    # whatever the ordering method applied, the first and second points need to be switched for this case
    # add a general condition to activate this switch
    u = boundary_pts[1] - boundary_pts[0]
    v = boundary_pts[2] - boundary_pts[1]
    cross_product = np.cross(u, v)
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    cos_theta = np.dot(u,v) / (magnitude_u * magnitude_v)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure sin_theta is within the valid range
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    print('angel12 = ', angle_degrees)
    reverse = False
    if angle_degrees > 90:
        start = boundary_pts[0]
        boundary_pts[0] = boundary_pts[1]
        boundary_pts[1] = start
        reverse = True
    boundary_pts = boundary_pts.tolist()
    boundary_pts.append(boundary_pts[0])
    offset_points = offset_points.tolist()
    offset_points.append(offset_points[0])
    if reverse:
        points_offset = np.vstack((np.array(offset_points), np.array(boundary_pts)))
    else:
        points_offset = np.vstack((np.array(boundary_pts), np.array(offset_points)))

    grid_offset = pv.StructuredGrid()
    grid_offset.points = points_offset
    grid_offset.dimensions = len(offset_points), 2, 1
    grid_offset = grid_offset.cast_to_unstructured_grid()
    grid_offset = grid_offset.extract_surface()
    return grid_offset, offset_points

# add Yoan post-process functions
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def verifier_ou_creer_dossier(chemin):
    if not os.path.exists(chemin):
        os.makedirs(chemin)
        print(f"Folder created : {chemin}")
    else:
        print(f"Folder already exist : {chemin}")

def compare_mesh_and_pointcloud(mesh_path: str, pointcloud_path: str) -> str:
    # Charger la mesh
    mesh = trimesh.load_mesh(mesh_path)
    # Calculer la moyenne des coordonnées z des sommets de la mesh
    mean_z_mesh = np.mean(mesh.vertices[:, 2])
    # Charger le nuage de points avec trimesh
    pointcloud = trimesh.load_mesh(pointcloud_path)
    # Calculer la moyenne des coordonnées z du nuage de points
    mean_z_pointcloud = np.mean(pointcloud.vertices[:, 2])
    # Comparaison des valeurs moyennes de z
    return "upper" if mean_z_pointcloud < mean_z_mesh else "lower"


def main(extracted=False):
    # for pipeline
    #die_save_path = './Die'
    #margin_save_path = './Margins'
    #cb_save_path = './MergingCbAndCrown/CrownBottoms_horizon'
    #offset_save_path = './RemovingShellBottom/Cases'

    # for personal test
    #die_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\Dies'
    #margin_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\ManualAdjustedMarginLines'
    #cb_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\CrownBottoms_generated_adjusted'
    #offset_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\MarginlineOffset_generated_adjusted'
    #margin_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\Marginlines'
    #cb_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\CrownBottoms_generated_test'
    #offset_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGenerationWithExtractedPrep\MarginlineOffset_generated_test'

    # test cases with die
    die_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGeneration\Die_GT'
    margin_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGeneration\Marginline_GT'
    cb_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGeneration\CrownBottoms_all_with_GT_new'
    offset_save_path = r'C:\Users\Ying\Documents\DentalAI\Datasets\Testcases\CrownBottomGeneration\marginlineOffsets_all_with_GT_new'

    # print('making path')
    if not os.path.exists(cb_save_path):
        os.mkdir(cb_save_path)
    #else:
        #clear_directory(cb_save_path)

    if not os.path.exists(offset_save_path):
        os.mkdir(offset_save_path)
    #else:
        #clear_directory(offset_save_path)
    #clear_directory(./Outputs)

    # in debug mode, intermediate files will be saved
    debug = True

    listMargins = [f for f in (os.listdir(margin_save_path)) if '.ply' in f or '.stl' in f]
    listFiles = [f for f in (os.listdir(die_save_path)) if '.ply' in f or '.stl' in f]
    # cases with offset issues
    #listFiles = ['8023-36_die_registered.ply', '8037-27_die_registered.ply', '8053-26_die_registered.ply',
    #             '8075-46_die_registered.ply', '8093-16_die_registered.ply', '8109-46_die_registered.ply',
    #             '8116-22_die_registered.ply']
    #listFiles = ['8109-46_die_registered.ply']
    listFiles = ['1700-11_die.stl']
    pattern = '*_die.stl'
    for entry in listFiles:
        if fnmatch.fnmatch(entry, pattern):
            start_time = time()
            die_file = entry
            print('find die file: ', die_file)
            case_id = die_file[0:7]
            die_path = os.path.join(die_save_path, die_file)
            marginline_file = next((element for element in listMargins if 'ply' or 'stl' in element), None)
            print('find marginline file: ', marginline_file)
            margin_path = os.path.join(margin_save_path, marginline_file)
            # Perform smoothing and save the result
            save_mesh_file = die_file[0:-4] + '_crownBottom.stl'
            save_offset_file = die_file[0:-4] + '_offsetMarginline.ply'
            # using mass center instead of die label to determine die_type
            die_type = compare_mesh_and_pointcloud(die_path, margin_path)
            print(die_type)
            die_mesh = pv.read(die_path)
            die_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            mean_die_normal = die_mesh['Normals'].mean(axis=0)
            # step1: cut die mesh
            offset_thickness = 0.25
            cut_mesh, flip = cutting_die(die_path, margin_path, die_type, extracted, thickness=offset_thickness)
            temp_file = 'die_cut.stl'
            cut_mesh.save(os.path.join(cb_save_path, temp_file))
            # merge close vertices
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(cb_save_path, temp_file))
            merge_percentage=5
            ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(merge_percentage))
            ms.apply_normal_point_cloud_smoothing(k=5)
            ms.save_current_mesh(os.path.join(cb_save_path, save_mesh_file))

            if debug:
                cut_die_file = 'step1_cut_die_mesh_' + case_id + '.stl'
                cut_mesh.save(os.path.join(cb_save_path, cut_die_file))

            # step2: block out undercut
            cut_mesh = pv.read(os.path.join(cb_save_path, save_mesh_file))
            blocked_cb = blockout_undercut(cut_mesh, blockout_start=0.1, cement_start=1.45)
            blocked_cb.save(os.path.join(cb_save_path, save_mesh_file))
            if debug:
                blocked_cb_file = 'step2_blockout_undercut.stl'
                blocked_cb.save(os.path.join(cb_save_path, blocked_cb_file))
            # step3 add cement gap and smoothing cut die
            cut_die_path = os.path.join(cb_save_path, save_mesh_file)
            smooth_file = 'step3_cement_and_smooth.stl'
            smoothed_path = os.path.join(cb_save_path, smooth_file)
            cement_and_smooth_cut_die(cut_die_path, smoothed_path, cement_start=1.45, cement_gap=0.035, blockout_distance=0.1, n_iter=10)
            extension_band, offset_marginline = generate_extension_band(die_path, smoothed_path, die_type, extracted, offset_thickness)
            final_cb = pv.read(smoothed_path)
            final_cb = final_cb.merge(extension_band)
            final_cb.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            if debug:
                expand_offset_file = 'step4_offset_band.stl'
                pv.save_meshio(os.path.join(cb_save_path, expand_offset_file), extension_band.triangulate())
                final_cb_file = 'step5_final_cb.stl'
                final_cb.save(os.path.join(cb_save_path, final_cb_file))

            mean_final_normal = final_cb['Normals'].mean(axis=0)
            # final crown bottom file name based on die_file
            final_cb.save(os.path.join(cb_save_path, save_mesh_file))

            if mean_die_normal[2] * mean_final_normal[2] > 0:
                # pyvista flip_normal doesn't work, try with pymeshlab
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(os.path.join(cb_save_path, save_mesh_file))
                ms.meshing_remove_duplicate_faces()
                ms.meshing_re_orient_faces_coherently()
                ms.meshing_invert_face_orientation()
                ms.save_current_mesh(os.path.join(cb_save_path, save_mesh_file))

            # smooth normal again for final cb
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(os.path.join(cb_save_path, save_mesh_file))
            ms.apply_normal_unsharp_mask_per_vertex(weight=0.5)
            ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(2))
            ms.meshing_re_orient_faces_coherently()
            ms.save_current_mesh(os.path.join(cb_save_path, save_mesh_file))

            # save offset_marginline
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(offset_marginline)
            o3d.io.write_point_cloud(os.path.join(offset_save_path, save_offset_file), pcd)
            # clean temporary file
            if not debug:
                os.remove(os.path.join(cb_save_path, smooth_file))
                os.remove(os.path.join(cb_save_path, temp_file))
            end_time = time()
            print('Generation completed in %6.3f seconds\n' % (end_time - start_time))
if __name__ == "__main__":
    main()