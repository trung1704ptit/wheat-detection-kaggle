import pandas as pd
import numpy as np
import cv2
import os
import re
import time
import datetime
from PIL import Image

# Albumentations is a Python library for image augmentation
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt

import numba
import cv2
# import ast
# from glob import glob
# from numba import jit
from typing import List, Union, Tuple

# Inputs required fro training
DIR_INPUT='dataset'
DIR_TRAIN=f'{DIR_INPUT}/train'
DIR_TEST=f'{DIR_INPUT}/test'

model_path ='fasterrcnn_resnet50_fpn.pth' # Path for the best model to be saved

es_patience = 2 # this is required for early stopping, the number of epochs we will wait no improvement before stopping

## Reading box coordinates form train.csv
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
print(train_df.shape)

train_df['x'] = -1
train_df['Y'] = -1
train_df['w'] = -1
train_df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x)) # find all float number for each row
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
train_df.drop(columns=['bbox'], inplace=True)
train_df['x'] = train_df['x'].astype(np.float)
train_df['y'] = train_df['y'].astype(np.float)
train_df['w'] = train_df['w'].astype(np.float)
train_df['h'] = train_df['h'].astype(np.float)

## Create training and validation datasets
image_ids = train_df['image_id'].unique() # collection all unique images
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]

valid_df = train_df[train_df['image_id'].isin(valid_ids)]  
train_df = train_df[train_df['image_id'].isin(train_ids)]

print(valid_df.shape, train_df.shape)

