import numpy as np
import os
import pyvista as pv
import time
import warnings


import open3d as o3d
from joblib import Parallel, delayed


def find_abnormal_normals_iterative_knn(pcd):
    """
    The full version requires the corresponding author to be contacted by email and a usage agreement signed
    """
    
def find_abnormal_normals_iterative_local_radius(pcd):
    """
    The full version requires the corresponding author to be contacted by email and a usage agreement signed

    """
   
def plot_pcd_normals(pcd):
    # Visualize the point cloud with normals using Open3D

    distances = pcd.compute_nearest_neighbor_distance()
    average_distance = np.mean(distances)
    print("Average distance between points: {:.4f}".format(average_distance))

    bbox=pcd.get_axis_aligned_bounding_box()
    bbox.scale(5, center=bbox.get_center())
    bbox.color=[1,0,0]

    #get the Y+ center of the bounding box
    top_center=bbox.get_center()
    top_center[1]=bbox.get_max_bound()[1]

    NormalPts=o3d.geometry.TriangleMesh.create_sphere(radius=np.mean(bbox.get_extent())/30)
    NormalPts.compute_vertex_normals()
    NormalPts.paint_uniform_color([0.1, 0.9, 0.1])
    NormalPts.translate(top_center)

    pcd.estimate_normals()
    #normalize the normal vectors
    pcd.normalize_normals()
    #orient the normal vectors
    #pcd.orient_normals_consistent_tangent_plane(100)
    pcd.orient_normals_towards_camera_location(camera_location=top_center)

    o3d.visualization.draw_geometries([pcd,bbox,NormalPts],
                                    point_show_normal=True)

def calculate_orientation(normals):
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / magnitudes

    dip_direction = np.mod(np.degrees(np.arctan2(normals[:, 1], normals[:, 0])), 360)

    normals[:, 2] = np.clip(normals[:, 2], -1, 1)
    dip = 90 - np.degrees(np.arcsin(np.abs(normals[:, 2])))

    return {'Dip': dip, 'Dip-Direction': dip_direction}