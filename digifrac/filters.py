import numpy as np
import os
import mcubes
import pyvista as pv
import fast_simplification
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import time
import warnings
import skimage as ski
from scipy.signal import fftconvolve
from tqdm import tqdm
import scipy as sp
import scipy.ndimage as spim
from skimage.measure import label, regionprops
import edt
from joblib import Parallel, delayed
import os
import vtk
import re

from .io import readVTKB, saveVTKB,cmd_execute

def trim_small_clusters(img,size=1000):
    ndim = img.ndim
    r=1
    rad = int(np.ceil(r))
    other = np.ones([2*rad + 1 for i in range(ndim)], dtype=bool)
    other[tuple(rad for i in range(ndim))] = False
    ball = edt.edt(other) <= r

    filtered_array = np.copy(img)
    labels, N = spim.label(filtered_array, structure=ball)
    print(f'Find {N} clusters:')

    id_sizes = np.array(spim.sum(img, labels, range(N + 1)))
    area_mask = id_sizes <= size
    print(f'Trimed {np.sum(area_mask)} clusters, left {len(id_sizes[~area_mask])} clusters')
    filtered_array[area_mask[labels]] = 0

    return filtered_array

def calculate_concentration(img_binary):
    fracture_data = img_binary  

    grid_size = tuple(s // 8 for s in fracture_data.shape)
    num_grids = (fracture_data.shape[0] // grid_size[0], 
                    fracture_data.shape[1] // grid_size[1], 
                    fracture_data.shape[2] // grid_size[2])

    grid_counts = np.zeros(num_grids)

    for i in range(num_grids[0]):
        for j in range(num_grids[1]):
            for k in range(num_grids[2]):
                x_start, x_end = i * grid_size[0], (i + 1) * grid_size[0]
                y_start, y_end = j * grid_size[1], (j + 1) * grid_size[1]
                z_start, z_end = k * grid_size[2], (k + 1) * grid_size[2]

                grid_counts[i, j, k] = np.sum(fracture_data[x_start:x_end, y_start:y_end, z_start:z_end])

    mean_density = np.mean(grid_counts)
    N = np.prod(num_grids)  
    n = 1  
    epsilon = 1e-10  

    concentration_value = 1 - (1 / N) * np.sum([x**(2 * n) / ((x**(2 * n) + (mean_density - x)**2) + epsilon)**n for x in grid_counts.flatten()])

    return concentration_value

def fractal_dimension(img_binary):
    def box_counting(data, box_sizes):
        counts = []
        for box_size in box_sizes:
            count = 0
            for x in range(0, data.shape[0], box_size):
                for y in range(0, data.shape[1], box_size):
                    for z in range(0, data.shape[2], box_size):
                        if np.any(data[x:min(x+box_size, data.shape[0]), 
                                        y:min(y+box_size, data.shape[1]), 
                                        z:min(z+box_size, data.shape[2])]):
                            count += 1
            counts.append(count)
        return counts

    min_dim = min(img_binary.shape) 
    box_sizes = [min_dim // (2 ** i) for i in range(int(np.log2(min_dim)) + 1) if min_dim // (2 ** i) > 0]

    counts = box_counting(img_binary, box_sizes)
    if len(counts) < 2:
        raise ValueError("The data does not have enough box counts for the provided sizes.")
    
    log_sizes = np.log(box_sizes).reshape(-1, 1)
    log_counts = np.log(counts)
    
    model = LinearRegression()
    model.fit(log_sizes, log_counts)
    coeffs = [model.coef_[0], model.intercept_]
    r_squared = model.score(log_sizes, log_counts)
    
    return -coeffs[0], box_sizes, counts, coeffs, r_squared

def calculate_fracture_complexity(img_binary):
    fractal_dimension_value, _, _, _, _ = fractal_dimension(img_binary)  
    concentration_value = calculate_concentration(img_binary)  
    fracture_complexity = fractal_dimension_value / np.sqrt(concentration_value)

    return fracture_complexity

def voxel_to_point_cloud(img_binary,return_pv = False):
    points = np.argwhere(img_binary > 0) 
    if return_pv:
        pv_cloud = pv.PolyData(points)
        return pv_cloud
    return points

def Hopkins_Skellam_Index(data):
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Input data must be a 2D array with three columns (x, y, z).")
    
    sample_size = int(data.shape[0] * 0.05)
    if sample_size < 1:
        raise ValueError("Sample size too small. Ensure the dataset has at least 20 points.")
    
    random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
    X_sample = data[random_indices]
    
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    uniform_random_sample = np.random.uniform(min_vals, max_vals, (sample_size, 3))
    
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    
    u_distances, _ = nbrs.kneighbors(uniform_random_sample, n_neighbors=1)
    u_distances = u_distances[:, 0]
    
    w_distances, _ = nbrs.kneighbors(X_sample, n_neighbors=2)
    w_distances = w_distances[:, 1]
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    H = u_sum / (u_sum + w_sum)
    
    return H

def calculate_surface_area(voxel_size, img_binary):
    surface_area = 0
    z_dim, y_dim, x_dim = img_binary.shape 

    for i in range(1, z_dim - 1):
        for j in range(1, y_dim - 1):
            for k in range(1, x_dim - 1):
                if img_binary[i, j, k] == 1:
                    exposed_faces = 0
                    if img_binary[i-1, j, k] == 0: exposed_faces += 1
                    if img_binary[i+1, j, k] == 0: exposed_faces += 1
                    if img_binary[i, j-1, k] == 0: exposed_faces += 1
                    if img_binary[i, j+1, k] == 0: exposed_faces += 1
                    if img_binary[i, j, k-1] == 0: exposed_faces += 1
                    if img_binary[i, j, k+1] == 0: exposed_faces += 1
                    
                    surface_area += exposed_faces
    
    surface_area_mm2 = surface_area * (voxel_size[0] * voxel_size[1])  
    
    return surface_area_mm2

def filter_with_dbscan(normals, eps=0.1, min_samples=10):
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / magnitudes

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(normals)

    deleted_indices = np.where(labels == -1)[0] 

    mask = labels != -1
    return normals[mask], labels, deleted_indices


def load_vti_files_as_numpy(input_dir="output_vti"):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory '{input_dir}' does not exist.")

    region_arrays = {}

    pattern = re.compile(r"region_(\d+)\.vti$")

    vti_files = []
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            region_id = int(match.group(1)) 
            vti_files.append((region_id, filename))

    vti_files.sort()

    for region_id, vti_file in vti_files:
        file_path = os.path.join(input_dir, vti_file)

        grid = pv.read(file_path)

        array_data = grid.point_data["Region"].reshape(grid.dimensions, order="F")

        array_data = array_data[:-1, :-1, :-1]

        region_arrays[region_id] = array_data

        print(f"Loaded {vti_file} (Region ID: {region_id}), shape: {array_data.shape}")

    return region_arrays
