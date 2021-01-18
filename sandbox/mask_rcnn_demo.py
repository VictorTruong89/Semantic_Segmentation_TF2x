import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath('../')

# Import Mask RCNN
sys.path.append(ROOT_DIR)   # TODO find local version of the library

import mask_rcnn_utils
import mask_rcnn_model as modellib
import mask_rcnn_visualize

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))    # TODO find local version
import mask_rcnn_coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Local path to the trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
# Download COCO trained weights from release if needed
# if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of the images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')


class InferenceConfig(coco.CocoConfig):
    # Set batch_size to 1 since we are running inference on 1 image at a time.
    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()