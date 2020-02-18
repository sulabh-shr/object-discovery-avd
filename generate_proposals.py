from detectron2_repo import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from datetime import datetime

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# ---------------------
import os
import glob
import torch
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model to use to generate the proposals. Must use full path of model corresponding to detectron2 repo.")
parser.add_argument("--output_dir", type=str, required=True, help="Root directory to store the outputs.")
parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
args = parser.parse_args()

# ------------------------ MODEL SELECTION AND CONFIGURATION ----------------------------------



# MODEL_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
MODEL_PATH = args.model
MODEL = MODEL_PATH.split('/')[1].split('.')[0]

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))

# Does not affect output bbox coordinates
cfg.INPUT.MAX_SIZE_TRAIN = 1333

# Setting to lower than 0.7 because using very low objectness
cfg.MODEL.RPN.NMS_THRESH = 0.5

# Objectness threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # set threshold for this model
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5     # Setting to lower than 0.7 because using very low objectness

# Setting to 500 for safety
cfg.TEST.DETECTIONS_PER_IMAGE = 150

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_PATH)
predictor = DefaultPredictor(cfg)

# -------------------------- FOLDER CREATION ----------------------------------
# OUTPUT_ROOT = '/mnt/sda2/workspace/self_supervised_outputs'
OUTPUT_ROOT = args.output_dir
RESULT_TYPE = 'predictor'
FOLDER_COUNT = 1
SLURM_SUFFIX = '_' + datetime.now().strftime("%d-%m-%Y_%H_%M_%S")  # By default, time

try:
	SLURM_JOB_ID = str(os.environ["SLURM_JOB_ID"])
	SLURM_SUFFIX = '_' + SLURM_JOB_ID
except KeyError:
	print('Slurm Job Id not avaialable')

folders_in_output_root = sorted([int(i[len(MODEL)+1:]) for i in os.listdir(OUTPUT_ROOT) if i.startswith(MODEL)])
if len(folders_in_output_root) > 0:
	FOLDER_COUNT = folders_in_output_root[-1] + 1

MODEL_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, f'{MODEL}_{FOLDER_COUNT}{SLURM_SUFFIX}')
PREDICTOR_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_PATH, RESULT_TYPE)
os.makedirs(PREDICTOR_OUTPUT_PATH)

# Save config file
with open(os.path.join(MODEL_OUTPUT_PATH, 'cfg.txt'), 'w') as f:
	f.write(cfg.dump())

with open(os.path.join(MODEL_OUTPUT_PATH, 'time.txt'), 'a') as f:
	f.write("START TIME: " + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'\n')

print(f'Model Output path = {MODEL_OUTPUT_PATH}')

# -------------------------- DATA PARAMS ----------------------------------
# DATASET_ROOT = '/mnt/sda2/workspace/DATASETS/ActiveVision'
DATASET_ROOT = args.dataset_root
SCENES = ['Home_001_1', 'Home_001_2', 'Home_002_1', 'Home_003_1', 
			'Home_003_2', 'Home_004_1', 'Home_004_2', 'Home_005_1',
            'Home_005_2', 'Home_006_1', 'Home_007_1', 'Home_008_1',
            'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1',
            'Home_014_2', 'Home_015_1', 'Home_016_1', 'Office_001_1']

# -------------------------- INFERENCE ----------------------------------

for scene in SCENES:
	image_paths = glob.glob(os.path.join(DATASET_ROOT, scene, 'jpg_rgb', '*.jpg'))
	print(f'Scene = {scene}, Number of Images = {len(image_paths)}')
	
	for image_path in tqdm(image_paths):
		
		image_file = os.path.basename(image_path)
		output_file = image_file.split('.')[0] + '.pt'

		img = cv2.imread(image_path)
		outputs = predictor(img)
		# Delete masks because they take too much space
		outputs["instances"].remove("pred_masks")
		torch.save(outputs, os.path.join(PREDICTOR_OUTPUT_PATH, output_file))
		del(outputs)

with open(os.path.join(MODEL_OUTPUT_PATH, 'time.txt'), 'a') as f:
	f.write("END TIME:   " + datetime.now().strftime("%m-%d-%Y_%H:%M:%S")+'\n')



# im = cv2.imread("input2.jpg")

# Rescale image
# width = int(im.shape[1] * scale_percent / 100)
# height = int(im.shape[0] * scale_percent / 100)
# dim = (width, height)
# im = cv2.resize(im, dim)

# outputs = predictor(im)

# Save predictions
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imwrite(f'o2_rpn_{cfg.MODEL.RPN.NMS_THRESH}_roi_{cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}_detperimg_{cfg.TEST.DETECTIONS_PER_IMAGE}_roiscore_{cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}.jpg', 
# 	v.get_image()[:,:,::-1])

# # print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(len(outputs["instances"].pred_boxes))
