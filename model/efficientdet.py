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
import imageio
import numpy as np
import tensorflow_hub as tfhub

model = tfhub.load("https://tfhub.dev/tensorflow/efficientdet/d3/1")

# return bounding box in shape of [ymin, xmin, ymax, xmax]
def get_box(filename):
    img = imageio.imread(filename, as_gray = False)
    img = np.reshape(img, [1, img.shape[0], img.shape[1], 3])
    prediction = model(img)

    if prediction["detection_boxes"].numpy().shape[1] > 0:
        return prediction["detection_boxes"].numpy()[0][0]
    else:
        return None

# return image with bounding box + save to current directory
def visualize_bbox(filename, bbox, save = True):
    img = imageio.imread(filename, as_gray = False)
    width = img.shape[1]
    height = img.shape[0]
    start_x = round(bbox[1] * width)
    start_y = round(bbox[0] * height)
    end_x = round(bbox[3] * width)
    end_y = round(bbox[2] * height)
    new_img = cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0,0,255), 2)
    if save:
        imageio.imwrite(filename.split(".")[0] + "_bb.jpg", new_img)
    return new_img
