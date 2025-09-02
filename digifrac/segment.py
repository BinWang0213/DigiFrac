import numpy as np
import os
import tifffile
from skimage import feature, future, img_as_ubyte
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from joblib import dump, load
from skimage.feature import multiscale_basic_features
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from tifffile import TiffFile,imwrite
from tqdm import tqdm  
import json
from PIL import Image, ImageDraw



class CTImageSegmenter:

    def load_data(file_path, shape=None, dtype=np.uint8, order='C'):
        """
        Load data from .tif or .raw file.

        Args:
            file_path (str): Path to the input file (.tif or .raw).
            shape (tuple, optional): Shape of the data (z, y, x) for 3D data (only for .raw).
            dtype (type, optional): Data type of the raw data (e.g., np.uint8, np.float32). Default is np.uint8.
            order (str, optional): Memory layout order ('C' for row-major, 'F' for column-major). Default is 'C'.

        Returns:
            np.ndarray: Loaded data as a NumPy array.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.tif':
            print("Loading .tif file...")
            data = tifffile.imread(file_path)
        elif ext == '.raw':
            if shape is None:
                raise ValueError("Shape must be provided for .raw files.")
            print("Loading .raw file...")
            data = np.fromfile(file_path, dtype=dtype)
            data = data.reshape(shape, order=order)
        else:
            raise ValueError("Unsupported file format. Supported formats are .tif and .raw.")

        print(f"Data loaded with shape {data.shape}.")
        return data

    def extract_single_slice(ct_data, slice_index, output_folder="training_data"):

        if slice_index < 0 or slice_index >= ct_data.shape[0]:
            raise ValueError(f"Invalid slice index! Must be between 0 and {ct_data.shape[0]-1}.")

        slice_data = ct_data[slice_index, :, :]

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"slice_{slice_index:03d}.tif")
        imwrite(output_path, slice_data)

        print(f"Slice {slice_index} extracted and saved to {output_path}")
        return slice_data

    def segment_ct_slice(ct_data, clf, slice_index, sigma_min, sigma_max):

        if slice_index < 0 or slice_index >= ct_data.shape[0]:
            raise ValueError(f"Slice index {slice_index} out of bounds (0, {ct_data.shape[0] - 1})")

        slice_data = ct_data[slice_index, :, :]
        print(f"Extracted slice {slice_index} with shape: {slice_data.shape}")

        features_func = partial(multiscale_basic_features,
                                intensity=True, edges=False, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max)

        features = features_func(slice_data)

        features_reshaped = features.reshape(-1, features.shape[-1])

        if features_reshaped.shape[1] != clf.n_features_in_:
            raise ValueError(
                f"Feature mismatch: {features_reshaped.shape[1]} features extracted, "
                f"but classifier expects {clf.n_features_in_} features"
            )

        predicted_labels = clf.predict(features_reshaped)

        segmented_result = predicted_labels.reshape(slice_data.shape)
        return segmented_result

    def train_segmenter(img_list, label_regions_list, sigma_min, sigma_max, model_path='segmenter_model.pkl', force_train=False):
        if not force_train and os.path.exists(model_path):
            clf = load(model_path)
            print("Loaded existing model.")
        else:
            clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)
            for img, label_regions in zip(img_list, label_regions_list):
                training_labels = np.zeros(img.shape[:2], dtype=np.uint8)
                for label, (y_min, y_max, x_min, x_max) in label_regions:
                    training_labels[y_min:y_max, x_min:x_max] = label

                features_func = partial(feature.multiscale_basic_features,
                                        intensity=True, edges=False, texture=True,
                                        sigma_min=sigma_min, sigma_max=sigma_max)
                features = features_func(img)

                clf = future.fit_segmenter(training_labels, features, clf)

            dump(clf, model_path)
            print(f"Model saved to {model_path}.")
        return clf

    def segment_3d_binary_and_save(ct_data, clf, sigma_min, sigma_max, output_path, output_folder="binary_segmented_slices"):
        os.makedirs(output_folder, exist_ok=True)
        segmented_slices = []

        for z in tqdm(range(ct_data.shape[0]), desc="Segmenting 3D CT Data"):
            slice_data = ct_data[z, :, :]

            features_func = partial(multiscale_basic_features,
                                    intensity=True, edges=False, texture=True,
                                    sigma_min=sigma_min, sigma_max=sigma_max)
            features = features_func(slice_data)
            features_reshaped = features.reshape(-1, features.shape[-1])
 
            predicted_labels = clf.predict(features_reshaped)
            segmented_slice = predicted_labels.reshape(slice_data.shape)

            binary_segmented_slice = np.zeros_like(segmented_slice, dtype=np.uint8)
            binary_segmented_slice[segmented_slice == 1] = 1

            segmented_slices.append(binary_segmented_slice)

            slice_output_path = os.path.join(output_folder, f"binary_segmented_slice_{z:03d}.tif")
            imwrite(slice_output_path, binary_segmented_slice.astype(np.uint8))

        segmented_3d_data = np.stack(segmented_slices, axis=0)

        imwrite(output_path, segmented_3d_data.astype(np.uint8))
        return segmented_3d_data

    def json_to_mask(json_file, image_shape):
        from PIL import Image, ImageDraw
        import json

        with open(json_file, 'r') as f:
            data = json.load(f)

        mask = np.zeros(image_shape, dtype=np.uint8)
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            points = [(float(p[0]), float(p[1])) for p in points]

            if label == "1":
                value = 1
            elif label == "2":
                value = 2
            else:
                continue  

            poly = Image.new("L", (image_shape[1], image_shape[0]), 0)
            ImageDraw.Draw(poly).polygon(points, outline=value, fill=value)
            mask += np.array(poly, dtype=np.uint8)

        return mask

    def generate_training_data_from_json(image_file, json_file, sigma_min, sigma_max):
        img = tifffile.imread(image_file)

        mask = CTImageSegmenter.json_to_mask(json_file, img.shape)

        features_func = partial(multiscale_basic_features,
                                intensity=True, edges=False, texture=True,
                                sigma_min=sigma_min, sigma_max=sigma_max)
        features = features_func(img)

        valid_mask = mask > 0
        features = features[valid_mask]
        labels = mask[valid_mask]

        return features, labels
    
    def train_segmenter_from_features(features_list, labels_list, model_path="segmenter_model.pkl"):

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        clf.fit(features, labels)

        dump(clf, model_path)
        print(f"Model saved to {model_path}")

        return clf
    
