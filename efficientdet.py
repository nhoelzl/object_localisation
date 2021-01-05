#!/usr/bin/env python3

# EfficientDet Object Detection
# * Single Shot Detector with EfficientNet-b3 + BiFPN feature extractor
# * Shared box predictor and focal loss
# * Trained on COCO 2017 dataset.
#
# 2021 (c) Micha Johannes Birklbauer
# 2021 (c) The TensorFlow Authors
#
# https://github.com/t0xic-m/
# micha.birklbauer@gmail.com

import cv2
import cvlib
import imageio
import numpy as np
import tensorflow_hub as tfhub

model = tfhub.load("https://tfhub.dev/tensorflow/efficientdet/d3/1")

# return bounding box in shape of [ymin, xmin, ymax, xmax]
def get_box(filename):
    img = imageio.imread(filename, as_gray=False)
    img = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
    prediction = model(img)

    if prediction["detection_boxes"].numpy().shape[1] > 0:
        return prediction["detection_boxes"].numpy()[0][0]

def visualize_bbox(filename, bbox):
    pass
