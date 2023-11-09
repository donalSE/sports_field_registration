from roboflow import Roboflow
import os
import pandas as pd

abs_dataset_directory = "/Users/donalconlon/Documents/GitHub/sports_field_registration/im_ge/training_data"

# Ensure the absolute directory exists, if not, create it
if not os.path.exists(abs_dataset_directory):
    os.makedirs(abs_dataset_directory)

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(7).download("coco-segmentation")

images_path = abs_dataset_directory = "/Users/donalconlon/Documents/GitHub/sports_field_registration/im_ge/training_data/train"

# Load the homography CSV file into a DataFrame
homography_df = pd.read_csv('/path/to/homograph.csv')

# Extract the list of filenames with valid homography from the DataFrame
valid_images = homography_df['frame_name'].tolist()

# Get all image filenames from the training directory
training_image_filenames = os.listdir(images_path)

# Filter out non-image files if necessary
training_image_filenames = [file for file in training_image_filenames if file.endswith('.jpg')]

# Iterate over the training image filenames
for image_filename in training_image_filenames:
    # Construct the absolute file path
    file_path = os.path.join(images_path, image_filename)
    # Check if the current file is not in the list of valid images
    if image_filename not in valid_images:
        # If the file is not in the list, delete it
        os.remove(file_path)
        print(f"Removed: {image_filename}")




