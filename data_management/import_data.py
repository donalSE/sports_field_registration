import numpy as np
import pandas as pd
import os

# Read data
images_path = './data_management/im_ge/training_data/train'
homography_csv_path = './homograph.csv'
homography_df = pd.read_csv(homography_csv_path)

def string_to_numpy(homography_string):
    # Remove unwanted characters and split by space to get individual elements
    cleaned_string = homography_string.replace('[', '').replace(']', '').replace(';', '').replace(',', '')
    # Convert string to a list of floats
    matrix_elements = [float(item) for item in cleaned_string.split()]
    # Convert list to a 3x3 numpy array
    return np.array(matrix_elements).reshape(3, 3)


# Assuming 'homography_df' is your DataFrame with the homography data
homography_df['homography_matrix'] = homography_df['homography_matrix'].apply(string_to_numpy)

# Invert homography mmatrix on every row
homography_df['homography_matrix'] = homography_df['homography_matrix'].apply(np.linalg.inv)

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

# Define the output path where the .npy files will be saved
output_npy_directory = './data_management/homography_matrices'
# change this to make dir if dir is not there
os.makedirs(output_npy_directory, exist_ok=True)

# Save each homography matrix as an .npy file
for index, row in homography_df.iterrows():
    homography_matrix = row['homography_matrix']
    if homography_matrix is not None:
        filename_without_extension = os.path.splitext(row['frame_name'])[0]
        npy_path = os.path.join(output_npy_directory, f"{filename_without_extension}.homography.npy")
        np.save(npy_path, homography_matrix)

print(f"Saved .npy files to {output_npy_directory}")
# print the first row of homography df
print(homography_df.iloc[0])




