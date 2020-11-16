#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:55:14 2019

@author: xingw
"""

import glob
import os
import shutil
from datetime import datetime

import sys
sys.setrecursionlimit(10000)

import cv2
import numpy as np
import tensorflow as tf
from keras.utils import plot_model

from keras.models import load_model
#from network.loss import DSSIM, DMSSSIM, DMSSSIML1, DMSSSIML2, MAEw
from network.network import get_model

# Params
flags = tf.app.flags  # @UndefinedVariable
flags.DEFINE_string("test_image_folder_path_LR", "/home/xingw/Dataset/McM_LR_2/*.png", "Path of the test image folder.")
flags.DEFINE_string("model_folder_path", "models/jdndmsr+_model.h5", "Path of the trained model folder.")
flags.DEFINE_integer("layers", 4, "number of Residual Groups.")
flags.DEFINE_integer("filters", 64, "number of filters of CNN.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_string("output_folder_path", "results/test_McM", "Path to directory to output files.")
flags.DEFINE_string("pixel_order", "rggb", "pixel oder for Bayer mosaic.")
flags.DEFINE_float("noise", 10.0, "standard deviation of the Gaussian noise added to the images.")
flags.DEFINE_integer("scale_factor", 2, "Scale factor 2, 3 or 4.")
FLAGS = flags.FLAGS

def bayer_mosaic(clean_image_content, pixel_order='rggb'):
    corrupt_image_content = clean_image_content.copy()
    image_height, image_width, _ = corrupt_image_content.shape
    mosaic_image_content = np.zeros((image_height, image_width))
    if pixel_order == 'rggb':
        mosaic_image_content[::2, ::2] = corrupt_image_content[::2, ::2, 2] # R
        mosaic_image_content[::2, 1::2] = corrupt_image_content[::2, 1::2, 1] # G
        mosaic_image_content[1::2, ::2] = corrupt_image_content[1::2, ::2, 1] # G
        mosaic_image_content[1::2, 1::2] = corrupt_image_content[1::2, 1::2, 0] # B
    elif pixel_order == 'bggr':
        mosaic_image_content[::2, ::2] = corrupt_image_content[::2, ::2, 0] # B
        mosaic_image_content[::2, 1::2] = corrupt_image_content[::2, 1::2, 1] # G
        mosaic_image_content[1::2, ::2] = corrupt_image_content[1::2, ::2, 1] # G
        mosaic_image_content[1::2, 1::2] = corrupt_image_content[1::2, 1::2, 2] # R
    elif pixel_order == 'grbg':
        mosaic_image_content[::2, ::2] = corrupt_image_content[::2, ::2, 1] # G
        mosaic_image_content[::2, 1::2] = corrupt_image_content[::2, 1::2, 2] # R
        mosaic_image_content[1::2, ::2] = corrupt_image_content[1::2, ::2, 0] # B
        mosaic_image_content[1::2, 1::2] = corrupt_image_content[1::2, 1::2, 1] # G
    elif pixel_order == 'gbrg':
        mosaic_image_content[::2, ::2] = corrupt_image_content[::2, ::2, 1] # G
        mosaic_image_content[::2, 1::2] = corrupt_image_content[::2, 1::2, 0] # B
        mosaic_image_content[1::2, ::2] = corrupt_image_content[1::2, ::2, 2] # R
        mosaic_image_content[1::2, 1::2] = corrupt_image_content[1::2, 1::2, 1] # G
    return np.expand_dims(mosaic_image_content, axis = 2)


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    test_image_folder_path_LR = FLAGS.test_image_folder_path_LR
    model_folder_path = FLAGS.model_folder_path
    layers = FLAGS.layers
    filters = FLAGS.filters
    batch_size = FLAGS.batch_size
    output_folder_path = FLAGS.output_folder_path
    noise_level = FLAGS.noise
    pixel_order = FLAGS.pixel_order
    scale_factor = FLAGS.scale_factor

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Initiating the model ...")
    if noise_level > 0:
        add_noise = True
    else:
        add_noise = False
    model = get_model("adam", "he_normal", "mean_absolute_error", filters, layers, scale_factor, add_noise)
    model.load_weights(model_folder_path)
    model.summary()
#    plot_model(model, to_file=os.path.join(output_folder_path, "model.png"), show_shapes=True, show_layer_names=True)

    print("Getting image file paths ...")
    test_image_file_path_LR_list = sorted(glob.glob(test_image_folder_path_LR))
    print("Predicting for each image in test set...")
    for image_file_path_LR in test_image_file_path_LR_list:
        # Read image
        img_name = image_file_path_LR.split('/')[-1].split('.')[0]
        image_content = cv2.imread(image_file_path_LR, 1)
        image_height, image_width, _ = image_content.shape
        if (image_height %2 !=0):
            image_height = image_height - 1
        if (image_width %2 != 0):
            image_width = image_width - 1
        image_content = image_content[:image_height, :image_width, :]
        
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, image_content.shape)
            corrupt_image_content = image_content.astype(np.float32) + noise.astype(np.float32)
            corrupt_image_content = np.clip(corrupt_image_content, 0, 255).astype(np.uint8)
            mosaic_image_content = bayer_mosaic(corrupt_image_content, pixel_order)
            mosaic_image_input = np.expand_dims(mosaic_image_content, axis = 0)
            estimate_noise = noise_level * np.ones((1, image_height // 2, image_width // 2, 1))
            mosaic_image_content_array, noise_array = np.array(mosaic_image_input, dtype=np.float32) / 255, np.array(estimate_noise, dtype=np.float32) / 255
            mosaic_image_content_array = (mosaic_image_content_array - 0.5) / 0.5
            predicted_image_content = model.predict([mosaic_image_content_array, noise_array], batch_size)
        else:
            mosaic_image_content = bayer_mosaic(image_content, pixel_order)
            mosaic_image_input = np.expand_dims(mosaic_image_content, axis = 0)
            mosaic_image_content_array = np.array(mosaic_image_input, dtype=np.float32) / 255
            mosaic_image_content_array = (mosaic_image_content_array - 0.5) / 0.5
            predicted_image_content = model.predict(mosaic_image_content_array, batch_size)
        
        predicted_image_content_HR = predicted_image_content[0]
        predicted_image_content_HR = np.clip(predicted_image_content_HR, -1, 1)

        cv2.imwrite(os.path.join(output_folder_path, "%s.png"%(img_name)), ((predicted_image_content_HR*0.5+0.5) * 255).astype(np.uint8))

if __name__ == "__main__":
    tf.app.run()
