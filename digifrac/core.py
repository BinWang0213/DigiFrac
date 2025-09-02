import numpy as np
import os

import pyvista as pv
import tifffile as tif


def load_img(file_path, shape=None, dtype=np.uint8, order='C'):

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.tif':
            print("Loading .tif file...")
            data = tif.imread(file_path)
        elif ext == '.raw':
            if shape is None:
                raise ValueError("Shape must be provided for .raw files.")
            print("Loading .raw file...")
            data = np.fromfile(file_path, dtype=dtype)
            data = data.reshape(shape, order=order)
        else:
            raise ValueError("Unsupported file format. Supported formats are .tif and .raw.")

        img = np.pad(data, pad_width=5, mode='constant', constant_values=0)

        print(f"Data loaded with shape {data.shape} and pad with 5 voxel.")
        return img

def save_img(image_data, filename, dtype=np.uint8):

    _, file_extension = os.path.splitext(filename)
    file_format_lower = file_extension.lower()

    if file_format_lower in [".tif", ".tiff"]:
        try:
            array_to_save = image_data
            if dtype is not None:
                array_to_save = image_data.astype(dtype)
            tif.imwrite(filename, array_to_save)
            print(f"3D array saved to '{filename}' as TIFF with dtype: {array_to_save.dtype}.")

        except Exception as e:
            raise Exception(f"Error saving TIFF file: {e}")

    elif file_format_lower == ".vti":
        try:
            array_to_save = image_data
            if dtype is not None:
                array_to_save = image_data.astype(dtype)

            dims = np.array(array_to_save.shape)+1
            grid = pv.ImageData(dimensions=dims)
            grid.spacing = (1, 1, 1) 
            grid.origin = (0, 0, 0)  
            grid.cell_data["ImageData"] = array_to_save.flatten(order='F') 

            grid.save(filename)
            print(f"3D array saved to '{filename}' as VTI with dtype: {array_to_save.dtype}.")

        except Exception as e:
            raise Exception(f"Error saving VTI file: {e}")
    else:
        raise ValueError(f"Unsupported file extension: '{file_extension}'. Filename must end with '.tif', '.tiff', or '.vti'.")
    

def img2pcd(img, scale=1.0, threshold=None,return_pv=False):

    if threshold is not None:
        img = img > threshold

    coords = np.argwhere(img)
    coords = coords * scale

    if return_pv:
        pcd = pv.PolyData(coords)
        return pcd
    else:
        return coords

def pcd2img(pcd, img_shape, scale=1.0):

    if isinstance(pcd, pv.PolyData):
        coords = pcd.points
    else:
        coords = pcd

    coords = coords / scale
    indices = np.round(coords).astype(int)

    img = np.zeros(img_shape, dtype=np.uint8)
    img[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    return img

def pcd2map(pcd):

    x_range = np.max(pcd[:,0])-np.min(pcd[:,0])
    y_range = np.max(pcd[:,1])-np.min(pcd[:,1])
    