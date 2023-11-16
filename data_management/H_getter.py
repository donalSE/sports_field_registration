# import csv
import os
import numpy as np
from pycocotools.coco import COCO
import json
from data_management.homo_utils import get_dst_points
from data_management.compute_homo import HomographyTransformer

# Define Paths
# COCO dataset path
ANNOTATION_FILE = './data_management/im_ge/train_annotations.coco.json'
IMAGE_DIR = './data_management/im_ge/train'
OUT_DIR = './data_management/homography_matrices'

# Define allowed classes
ALLOWED_CLASSES = ['0A', '0B', '1A', '1B', '1C', '1D', '1E', '1F', '1G', '1H', '1I', '1J', '1K', '1L', '1M', '1N', '1O',
                   '1P', '1Q', '1R', '1S', '1T', '2A', '2B', '2C', '2D', '2E', '2F', '2G', '2H', '2I', '2J', '2K', '2L',
                   '2M', '2N', '2O', '2P', '2Q', '2R', '2S', '2T', '1GPA', '1GPB', '2GPA', '2GPB']

# 1. Import Data, and transform to kp's
# --------------------------------------

# Load your COCO annotations file
with open(ANNOTATION_FILE, 'r') as f:
    coco_data = json.load(f)

# Create a mapping from class ID to class name
id_to_name_mapping = {category['id']: category['name'] for category in coco_data['categories']}
coco = COCO(ANNOTATION_FILE)
image_ids = coco.getImgIds()
allowed_class_ids = coco.getCatIds(catNms=ALLOWED_CLASSES)

# Load images and annotations
data_list = []
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=allowed_class_ids)
    annotations = coco.loadAnns(ann_ids)

    # Update class IDs in annotations to class names using the mapping
    for ann in annotations:
        class_id = ann['category_id']
        class_name = id_to_name_mapping[class_id]
        ann['category_id'] = class_name  # Update class ID to class name

        # Calculate the center of the bounding box
        bbox = ann['bbox']  # bbox format is [x, y, width, height]
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2

        # Replace the bbox with the center point
        ann['keypoint'] = [center_x, center_y]

    data_list.append({
        'image_id': img_id,
        'file_name': img_info['file_name'],
        'annotations': annotations,
    })

print(data_list[0])
reformatted_data_list = []
# Process each entry in the data list
for data in data_list:
    # Create a dictionary for the current image
    image_data = {
        'image_id': data['image_id'],
        'file_name': data['file_name'],
        'keypoints': []
    }

    # Process each annotation in the current image
    for annotation in data['annotations']:
        # Extract the keypoint and its category
        keypoint = {
            'category_id': annotation['category_id'],
            'keypoint': annotation['keypoint']
        }
        # Add the keypoint to the list of keypoints in the image data
        image_data['keypoints'].append(keypoint)

    # Add the image data to the reformatted data list
    reformatted_data_list.append(image_data)

print(reformatted_data_list[0])

# 2. Extract H matrix for each image. Also save to df for future investigation
# --------------------------------------
homographies_per_image = {}  # Dictionary to store homography matrices
dst_mapping = get_dst_points()

csv_file_path = os.path.join(OUT_DIR, 'homography_check.csv')

# # Create a CSV file to write the homography matrices, pixel points, and real-world points
# with open(csv_file_path, 'w', newline='') as csvfile:
#     fieldnames = ['file_name', 'homography_matrix', 'pixel_points', 'real_world_points']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

# Process each entry in the data list
for image_data in reformatted_data_list:
    pixel_points = []
    real_world_points = []

    # Get the corresponding real-world points for each keypoint
    for detection in image_data['keypoints']:
        pixel_point = detection['keypoint']
        category_id = detection['category_id']
        real_world_point = dst_mapping.get(category_id)

        if real_world_point:
            pixel_points.append(pixel_point)
            real_world_points.append(real_world_point)

    # Check if we have enough points to compute the homography
    if len(pixel_points) >= 4 and len(real_world_points) >= 4:
        # Compute the homography matrix
        transformer = HomographyTransformer(pixel_points, real_world_points)
        H_matrix = transformer.get_homography_matrix()
        homographies_per_image[image_data['image_id']] = H_matrix

        # Prepare pixel and real-world points for CSV
        pixel_points_str = ';'.join([f"({x[0]}, {x[1]})" for x in pixel_points])
        real_world_points_str = ';'.join([f"({x[0]}, {x[1]})" for x in real_world_points])
        homography_str = ';'.join([';'.join([f"{value}" for value in row]) for row in H_matrix])

        # # Write to CSV
        # writer.writerow({
        #     'file_name': image_data['file_name'],
        #     'homography_matrix': homography_str,
        #     'pixel_points': pixel_points_str,
        #     'real_world_points': real_world_points_str
        # })
    else:
        homographies_per_image[image_data['image_id']] = None

# 3. Save H matrix for each image to its own .npy file
# --------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
# Save each homography matrix by file name
for image_data in reformatted_data_list:
    # Retrieve the homography matrix using the image_id
    H = homographies_per_image.get(image_data['image_id'])
    if H is not None:
        # Split the file name at underscores
        parts = image_data['file_name'].split('_')
        # Reconstruct the filename using the first two parts and replace the rest with '.jpg'
        base_name = '_'.join(parts[:2])
        homography_file_name = f"{base_name}_homography.npy"
        homography_path = os.path.join(OUT_DIR, homography_file_name)

        # Save the homography matrix to a .npy file
        np.save(homography_path, H)
        print(f"Saved homography matrix for file {homography_file_name}")


# 4. Rename Image files to match homographies
# --------------------------------------
for filename in os.listdir(IMAGE_DIR):
    # Check if the file is a JPEG image
    if filename.endswith('.jpg'):
        # Split the filename at underscores and take the first two parts
        parts = filename.split('_')
        if len(parts) > 2:
            # Create the new filename using the first two parts and '.jpg'
            new_filename = '_'.join(parts[:2]) + '.jpg'
            # Construct full file paths
            old_file = os.path.join(IMAGE_DIR, filename)
            new_file = os.path.join(IMAGE_DIR, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f"Renamed '{filename}' to '{new_filename}'")
