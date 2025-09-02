import numpy as np
import os
import pyvista as pv
import time
import warnings
import skimage as ski

import edt
from joblib import Parallel, delayed


from .io import readVTKB, saveVTKB,cmd_execute
from .core import save_img


def extract_surface_skeleton(img_data,engine=["IMA3D","-g 3"], is_clean=True):
    #generate a skeleton from a 3D image using Converage Axis++
    #img_data: 3D numpy array
    '''
    Extract Medial Axis surface using voxel based thinning method

    Voxel method library: http://www.staff.science.uu.nl/~telea001/pmwiki.php/Shapes/SkelBenchmark

    Voxel core could be tested as well, we will try it later

    Integer medial axis (IMA3D): https://github.com/BinWang0213/3rdParty-IMA3D
        [input]: binary UCHAR VTK image
        [args]: IMA3D.exe in.vtk -g gamma
            gamma: thinning paramters, higher value thinner volume left
        [output]: binary numpy image
    '''

    if not os.path.exists("Results"): os.makedirs('Results')
    pwd=os.getcwd()
    output_dir=os.path.join(pwd,'Results','')
    library_root_dir = os.path.dirname(os.path.abspath(__file__))

    img_skeleton=None

    if("IMA3D" in engine):#Integer medial axis engine
        #1. Output image to binary VTK file format
        input_path=os.path.join(output_dir,'ThinningInput_IMA3D.vtk')
        saveVTKB(input_path,img_data)

        #2. Run engine to thinning
        args=engine[1]
        ouput_path=os.path.join(output_dir,'ThinningOutput_IMA3D.vtk')

        path_to_engine = os.path.join(library_root_dir, 'IMA3D','IMA3D.exe')

        cmd=path_to_engine + ' '+ input_path + " " + args
        cmd+=' -o ' + ouput_path
        print("[Image] Running Integer medial axis (IMA3D) image thinning algorithm @\n \t",cmd)

        info=cmd_execute(cmd)

        #3. Load skeleton back
        engine_dir=os.path.dirname(path_to_engine)
        img_skeleton=readVTKB(ouput_path)

        #IMA3D has a bug needs shift image 1 voxel in x direction
        img_skeleton[0,:,:]=img_skeleton[1,:,:]
        img_skeleton=np.roll(img_skeleton,-1,axis=0)

    return img_skeleton
def skeletonize_thin(img_data):
    """
    The full version requires the corresponding author to be contacted by email and a usage agreement signed
    """

def split_image_into_eighths(img):

    z, y, x = img.shape
    z_mid, y_mid, x_mid = z // 2, y // 2, x // 2
    sub_images = [
        img[:z_mid, :y_mid, :x_mid],
        img[:z_mid, :y_mid, x_mid:],
        img[:z_mid, y_mid:, :x_mid],
        img[:z_mid, y_mid:, x_mid:],
        img[z_mid:, :y_mid, :x_mid],
        img[z_mid:, :y_mid, x_mid:],
        img[z_mid:, y_mid:, :x_mid],
        img[z_mid:, y_mid:, x_mid:]
    ]
    return sub_images

def combine_sub_images(sub_images, original_shape):

    z, y, x = original_shape
    z_mid, y_mid, x_mid = z // 2, y // 2, x // 2
    
    combined_image = np.zeros(original_shape, dtype=sub_images[0].dtype)
    combined_image[:z_mid, :y_mid, :x_mid] = sub_images[0]
    combined_image[:z_mid, :y_mid, x_mid:] = sub_images[1]
    combined_image[:z_mid, y_mid:, :x_mid] = sub_images[2]
    combined_image[:z_mid, y_mid:, x_mid:] = sub_images[3]
    combined_image[z_mid:, :y_mid, :x_mid] = sub_images[4]
    combined_image[z_mid:, :y_mid, x_mid:] = sub_images[5]
    combined_image[z_mid:, y_mid:, :x_mid] = sub_images[6]
    combined_image[z_mid:, y_mid:, x_mid:] = sub_images[7]
    
    return combined_image

def process_image_in_chunks(img):

    sub_images = split_image_into_eighths(img)
    skel_sub_images = []
    mask_sub_images = []
    skel_slice_sub_images = []

    for sub_img in sub_images:
        img_skel_slice, img_skels = skeletonize_thin(sub_img, debug=True)
        img_skel_slice, img_skels = skeletonize_thin(img_skel_slice > 0, debug=True)
        img_skel_slice, img_skels = skeletonize_thin(img_skel_slice > 0, debug=True)
        
        skel_x, skel_y, skel_z, skel_comm = img_skels
        img_skel = np.logical_or(skel_comm, skel_z)
        mask_skel = img_skel > 0  

        skel_sub_images.append(img_skel)
        mask_sub_images.append(mask_skel)
        skel_slice_sub_images.append(img_skel_slice)

    combined_skeleton = combine_sub_images(skel_sub_images, img.shape)
    combined_mask = combine_sub_images(mask_sub_images, img.shape)
    combined_skel_slice = combine_sub_images(skel_slice_sub_images, img.shape)  # 确保它被计算

    # print(f"Returning: {type(combined_skel_slice)}, {type(combined_skeleton)}, {type(combined_mask)}")
    return combined_skel_slice, combined_skeleton, combined_mask

def save_vtk_results(img, img_skel_slice, img_skel, filename="Results/image_info.vti"):

    origin_filename = filename.replace(".vti", "_origin.vti")
    save_img(img, origin_filename)

    img_info = pv.read(origin_filename)
    img_info['img'] = img.flatten(order='F')
    img_info['skel_slice'] = img_skel_slice.flatten(order='F')
    img_info['skel'] = img_skel.flatten(order='F')

    img_info.save(filename)


