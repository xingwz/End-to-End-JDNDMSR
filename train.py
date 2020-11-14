#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07

@author: Wenzhu, Xing
wenzhu.xing@tuni.fi
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob
import multiprocessing
import os
import pickle
import shutil
import sys
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.utils import plot_model
import random

from network.network import get_model
from load_dataset import DataSequence

from keras.models import load_model

# Params
flags = tf.app.flags  # @UndefinedVariable
flags.DEFINE_list("train_image_folder_path_LR", ["/home/Dataset/DIV2K_train_LR_2/{}.png","/home/Dataset/Flickr2K_LR_2/{}.png"], "Path of the train image folder.")
flags.DEFINE_string("valid_image_folder_path_LR", "/home/Dataset/DIV2K_valid_LR_2/{}.png", "Path of the valid image folder.")
flags.DEFINE_list("test_image_folder_path_LR", ["/home/Dataset/McM_LR_2/*.png","/home/Dataset/kodak_LR_2/*.png","/home/Dataset/B100_LR_2/*.png","/home/Dataset/urban100_LR_2/*.png"], "Path of the test image folder.")
flags.DEFINE_list("train_image_folder_path", ["/home/Dataset/DIV2K_train_HR/{}.png","/home/Dataset/Flickr2K_HR/{}.png"], "Path of the train image folder.")
flags.DEFINE_string("valid_image_folder_path", "/home/Dataset/DIV2K_valid_HR/{}.png", "Path of the valid image folder.")
flags.DEFINE_list("test_image_folder_path", ["/home/Dataset/McM/*.png","/home/Dataset/kodak/*.png","/home/Dataset/B100/*.png","/home/Dataset/urban100/*.png"], "Path of the test image folder.")
flags.DEFINE_string("train_img_id", "*", "select image id of train data")
flags.DEFINE_string("valid_img_id", "*", "select image id of valid data")
flags.DEFINE_integer("image_width", 64, "Width of the images.")
flags.DEFINE_integer("image_height", 64, "Height of the images.")
flags.DEFINE_integer("layers", 4, "number of Residual Groups.")
flags.DEFINE_integer("filters", 64, "number of filters of CNN.")
flags.DEFINE_integer("batch_size", 16, "Batch size.")
flags.DEFINE_integer("inspection_size", 4, "Inspection size.")
flags.DEFINE_string("learning_rate_mode", "half-cut", "Mode of the learning rate scheduler.")  # ["constant", "linear", "cosine"]
flags.DEFINE_float("learning_rate_start", 0.001, "Starting learning rate.")
flags.DEFINE_float("learning_rate_end", 0.0001, "Ending learning rate.")
flags.DEFINE_integer("train_steps_per_epoch", 2000, "Number of steps per epoch for training.")
flags.DEFINE_integer("valid_steps_per_epoch", 200, "Number of steps per epoch for validation.")
flags.DEFINE_integer("epoch_num", 100, "Number of epochs.")
flags.DEFINE_integer("worker_num", multiprocessing.cpu_count(), "Number of processes to spin up for data generator.")
flags.DEFINE_string("output_folder_path", "/home/results", "Path to directory to output files.")
flags.DEFINE_string("optimizer", "adam", "optimizer of model")
flags.DEFINE_string("initializer", "he_normal", "initializer of model")
flags.DEFINE_integer("scale_factor", 2, "Scale factor 2, 3 or 4.")
flags.DEFINE_float("noise", 0.0784, "standard deviation of the Gaussian noise added to the images.")
flags.DEFINE_string("loss_function", "mean_absolute_error", "loss function of training.")
FLAGS = flags.FLAGS


def learning_rate_scheduler(epoch, learning_rate_mode, learning_rate_start, learning_rate_end, epoch_num):
    learning_rate = None
    if learning_rate_mode == "constant":
        assert learning_rate_start == learning_rate_end, "starting and ending learning rates should be equal!"
        learning_rate = learning_rate_start
    elif learning_rate_mode == "linear":
        learning_rate = (learning_rate_end - learning_rate_start) / epoch_num * epoch + learning_rate_start
    elif learning_rate_mode == "half-cut":
        if epoch < 10:
            learning_rate = learning_rate_start
        elif epoch < 100:
            learning_rate = learning_rate_start / 10
        else:
            learning_rate = learning_rate_start / 20
    elif learning_rate_mode == "constant-linear":
        if epoch < 50:
            learning_rate = learning_rate_start
        else:
            learning_rate = (learning_rate_end - learning_rate_start) / (epoch_num -50) * (epoch - 49) + learning_rate_start
    elif learning_rate_mode == "cosine":
        assert learning_rate_start > learning_rate_end, "starting learning rate should be higher than ending learning rate!"
        learning_rate = (learning_rate_start - learning_rate_end) / 2 * np.cos(np.pi * epoch / (epoch_num - 1)) + (learning_rate_start + learning_rate_end) / 2
    else:
        assert False, "{} is an invalid argument!".format(learning_rate_mode)
    assert learning_rate > 0, "learning_rate {} is not positive!".format(learning_rate)
    return learning_rate


class HistoryLogger(Callback):
    def __init__(self, output_folder_path):
        super(HistoryLogger, self).__init__()

        self.accumulated_logs_dict = {}
        self.output_folder_path = output_folder_path

    def visualize(self, loss_name):
        # Unpack the values
        epoch_to_loss_value_dict = self.accumulated_logs_dict[loss_name]
        epoch_list = sorted(epoch_to_loss_value_dict.keys())
        loss_value_list = [epoch_to_loss_value_dict[epoch] for epoch in epoch_list]

        # Save the figure to disk
        figure = plt.figure()
        if isinstance(loss_value_list[0], dict):
            for metric_name in loss_value_list[0].keys():
                metric_value_list = [loss_value[metric_name] for loss_value in loss_value_list]
                print("{} {} {:.6f}".format(loss_name, metric_name, metric_value_list[-1]))
                plt.plot(epoch_list, metric_value_list, label="{} {:.6f}".format(metric_name, metric_value_list[-1]))
        else:
            plt.plot(epoch_list, loss_value_list, label="{} {:.6f}".format(loss_name, loss_value_list[-1]))
        plt.grid(True)
        plt.legend(loc="best")
        plt.savefig(os.path.join(self.output_folder_path, "{}.png".format(loss_name)))
        plt.close(figure)

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        # Visualize each figure
        for loss_name, loss_value in logs.items():
            if loss_name not in self.accumulated_logs_dict:
                self.accumulated_logs_dict[loss_name] = {}
            self.accumulated_logs_dict[loss_name][epoch] = loss_value
            self.visualize(loss_name)

        # Save the accumulated_logs_dict to disk
        with open(os.path.join(self.output_folder_path, "accumulated_logs_dict.pkl"), "wb") as file_object:
            pickle.dump(self.accumulated_logs_dict, file_object, pickle.HIGHEST_PROTOCOL)

class Inspection(Callback):
    def __init__(self, inspection_generator, output_folder_path, add_noise):
        super(Inspection, self).__init__()

        self.inspection_generator = inspection_generator
        self.output_folder_path = output_folder_path
        self.add_noise = add_noise
        self.format_image_content = lambda image_content: ((image_content * 0.5 + 0.5) * 255).astype(np.uint8)

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        if (epoch+1) % 10 == 1:
            image_and_noise_list, clean_image_content_array = self.inspection_generator[0]
            if self.add_noise:
                corrupt_image_content_array = image_and_noise_list[0]
            else:
                corrupt_image_content_array = image_and_noise_list
            self.inspection_generator.on_epoch_end()
            predicted_image_content_array = self.model.predict_on_batch(image_and_noise_list)
            predicted_image_content_array = np.clip(predicted_image_content_array, -1, 1)
    
            subfolder_path = os.path.join(self.output_folder_path, "epoch{:04d}".format(epoch + 1))
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
    
            for sample_index, (corrupt_image_content, clean_image_content, predicted_image_content) in enumerate(zip(corrupt_image_content_array, clean_image_content_array, predicted_image_content_array), start=1):
                cv2.imwrite(os.path.join(subfolder_path, "epoch_{}_sample_{}_corrupt.png".format(epoch + 1, sample_index)), self.format_image_content(corrupt_image_content))
                cv2.imwrite(os.path.join(subfolder_path, "epoch_{}_sample_{}_clean.png".format(epoch + 1, sample_index)), self.format_image_content(clean_image_content))
                cv2.imwrite(os.path.join(subfolder_path, "epoch_{}_sample_{}_predicted.png".format(epoch + 1, sample_index)), self.format_image_content(predicted_image_content))


def apply_workaround_of_OOM(data_generator):
    while True:
        for data_tuple in data_generator:
            yield data_tuple
        data_generator.on_epoch_end()


def main(_):
    print("Getting hyperparameters ...")
    print("Using command {}".format(" ".join(sys.argv)))
    flag_values_dict = FLAGS.flag_values_dict()
    for flag_name in sorted(flag_values_dict.keys()):
        flag_value = flag_values_dict[flag_name]
        print(flag_name, flag_value)
    train_image_folder_path_LR = FLAGS.train_image_folder_path_LR
    valid_image_folder_path_LR = FLAGS.valid_image_folder_path_LR
    test_image_folder_path_LR = FLAGS.test_image_folder_path_LR
    train_image_folder_path = FLAGS.train_image_folder_path
    valid_image_folder_path = FLAGS.valid_image_folder_path
    test_image_folder_path = FLAGS.test_image_folder_path
    image_width = FLAGS.image_width
    image_height = FLAGS.image_height
    filters = FLAGS.filters
    batch_size = FLAGS.batch_size
    inspection_size = FLAGS.inspection_size
    learning_rate_mode, learning_rate_start, learning_rate_end = FLAGS.learning_rate_mode, FLAGS.learning_rate_start, FLAGS.learning_rate_end
    train_steps_per_epoch = FLAGS.train_steps_per_epoch
    valid_steps_per_epoch = FLAGS.valid_steps_per_epoch
    epoch_num = FLAGS.epoch_num
    worker_num = FLAGS.worker_num
    output_folder_path = FLAGS.output_folder_path
    optimizer = FLAGS.optimizer
    initializer = FLAGS.initializer
    layers = FLAGS.layers
    scale_factor = FLAGS.scale_factor
    max_noise = FLAGS.noise
    loss_function = FLAGS.loss_function

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Initiating the model ...")
    if max_noise > 0:
        add_noise = True
    else:
        add_noise = False
    
    model = get_model(optimizer, initializer, loss_function, filters, layers, scale_factor)
    model.summary()
    plot_model(model, to_file=os.path.join(output_folder_path, "model.png"), show_shapes=True, show_layer_names=True)

    print("Getting image file paths ...")
    Flickr2K_LR_file_path_list = sorted(glob.glob((train_image_folder_path_LR[1]).format(FLAGS.train_img_id)))
    shuffle_index = random.sample(range(len(Flickr2K_LR_file_path_list)),len(Flickr2K_LR_file_path_list))
    train_index = shuffle_index[:(len(Flickr2K_LR_file_path_list)-100)]
    valid_index = shuffle_index[-100:]
    train_Flickr2K_LR_file_path_list = [Flickr2K_LR_file_path_list[i] for i in train_index] 
    valid_Flickr2K_LR_file_path_list = [Flickr2K_LR_file_path_list[i] for i in valid_index] 
    train_image_file_path_LR_list = sorted(glob.glob((train_image_folder_path_LR[0]).format(FLAGS.train_img_id))+train_Flickr2K_LR_file_path_list)
    valid_image_file_path_LR_list = sorted(glob.glob((valid_image_folder_path_LR).format(FLAGS.valid_img_id))+valid_Flickr2K_LR_file_path_list)
    test_image_file_path_LR_list = sorted(glob.glob(test_image_folder_path_LR[0])+glob.glob(test_image_folder_path_LR[1])+glob.glob(test_image_folder_path_LR[2])+glob.glob(test_image_folder_path_LR[3]))
#    test_image_file_path_LR_list = sorted(glob.glob(test_image_folder_path_LR[0]))
    Flickr2K_HR_file_path_list = sorted(glob.glob((train_image_folder_path[1]).format(FLAGS.train_img_id)))
    train_Flickr2K_HR_file_path_list = [Flickr2K_HR_file_path_list[i] for i in train_index] 
    valid_Flickr2K_HR_file_path_list = [Flickr2K_HR_file_path_list[i] for i in valid_index] 
    train_image_file_path_list = sorted(glob.glob((train_image_folder_path[0]).format(FLAGS.train_img_id))+train_Flickr2K_HR_file_path_list)
    valid_image_file_path_list = sorted(glob.glob((valid_image_folder_path).format(FLAGS.valid_img_id))+valid_Flickr2K_HR_file_path_list)
    test_image_file_path_list = sorted(glob.glob(test_image_folder_path[0])+glob.glob(test_image_folder_path[1])+glob.glob(test_image_folder_path[2])+glob.glob(test_image_folder_path[3]))

    
    print("Initiating the data generators ...")
    train_generator = DataSequence(train_image_file_path_LR_list, train_image_file_path_list, image_height, image_width, batch_size, train_steps_per_epoch, scale_factor, max_noise)
    valid_generator = DataSequence(valid_image_file_path_LR_list, valid_image_file_path_list, image_height, image_width, batch_size, valid_steps_per_epoch, scale_factor, max_noise)
    valid_generator = apply_workaround_of_OOM(valid_generator)
    inspection_generator = DataSequence(test_image_file_path_LR_list, test_image_file_path_list, image_height, image_width, min(batch_size, inspection_size), 1, scale_factor, max_noise)

    print("Perform training ...")
    modelcheckpoint_callback = ModelCheckpoint(filepath=os.path.join(output_folder_path, "model.h5"), monitor="val_loss", mode="min",
                                               save_best_only=True, save_weights_only=False, verbose=1)
    learningratescheduler_callback = LearningRateScheduler(schedule=lambda epoch: learning_rate_scheduler(
        epoch, learning_rate_mode, learning_rate_start, learning_rate_end, epoch_num), verbose=1)
    historylogger_callback = HistoryLogger(output_folder_path)
    inspection_callback = Inspection(inspection_generator, output_folder_path, add_noise)
    csv_logger = CSVLogger(os.path.join(output_folder_path, "log.csv"), append=True, separator=';')
    use_multiprocessing = worker_num > 1
    model.fit_generator(generator=train_generator, steps_per_epoch=train_steps_per_epoch,
                        validation_data=valid_generator, validation_steps=valid_steps_per_epoch,
                        callbacks=[modelcheckpoint_callback, learningratescheduler_callback, historylogger_callback, inspection_callback, csv_logger],
                        epochs=epoch_num, workers=worker_num, use_multiprocessing=use_multiprocessing, verbose=1)

    print("All done!")


if __name__ == "__main__":
    tf.app.run()
