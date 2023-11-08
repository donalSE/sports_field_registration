from roboflow import Roboflow
import os

abs_dataset_directory = "/Users/donalconlon/Documents/GitHub/sports_field_registration/im_ge/training_data"

# Ensure the absolute directory exists, if not, create it
if not os.path.exists(abs_dataset_directory):
    os.makedirs(abs_dataset_directory)

rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("donals-thesis").project("football-id-2")
dataset = project.version(7).download("coco-segmentation")