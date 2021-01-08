"""
Created on 30.12.2020, by Nicole Hölzl
"""

# import packages
import tensorflow as tf
import tensorflow_hub as hub

# TODO:
    # Search for trainable models in Tensorflow
    # Fix tensorflow-hub issue

    # Object Detection:
    # https://www.tensorflow.org/hub/tutorials/object_detection
    # https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/

    # Trainable Classifier (without Localisation);
    # https://hub.tensorflow.google.cn/tensorflow/efficientnet/b7/classification/1
    # https://hub.tensorflow.google.cn/tensorflow/resnet_50/classification/1
    # https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/classification/4
    # https://hub.tensorflow.google.cn/tensorflow/efficientnet/b0/classification/1


# test loading efficientdet model 
# module_spec = hub.load_module_spec("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b3/classification/1")
# height, width = hub.get_expected_image_size(module_spec)
# images = ...  # A batch of images with shape [batch_size, height, width, 3].
print('loading model...')
# module = hub.Module("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b3/classification/1", trainable=True)
# module = hub.Module(module_spec)
# m = tf.keras.Sequential([
#    hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/efficientnet/b7/classification/1")
# ])
print('model successfully loaded!')
# logits = module(images)   # A batch with shape [batch_size, num_classes].






