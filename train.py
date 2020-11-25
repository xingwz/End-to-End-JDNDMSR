#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07

@author: Wenzhu, Xing
wenzhu.xing@tuni.fi
"""
import matplotlib
matplotlib.use("Agg")

import glob
import multiprocessing
import os
import shutil
import sys
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger
from keras.utils import plot_model
import random

from network.network import get_model
from load_dataset import DataSequence
from utils import learning_rate_scheduler, apply_workaround_of_OOM, HistoryLogger, Inspection

from keras.models import load_model

# Params
flags = tf.app.flags  # @UndefinedVariable
flags.DEFINE_list("train_image_folder_path_LR", ["data/DIV2K_train_LR_2/{}.png","data/Flickr2K_LR_2/{}.png"], "Path of the train image folder.")
flags.DEFINE_string("valid_image_folder_path_LR", "data/DIV2K_valid_LR_2/{}.png", "Path of the valid image folder.")
flags.DEFINE_list("test_image_folder_path_LR", ["data/McM_LR_2/*.png","data/kodak_LR_2/*.png","data/B100_LR_2/*.png","data/urban100_LR_2/*.png"], "Path of the test image folder.")
flags.DEFINE_list("train_image_folder_path", ["data/DIV2K_train_HR/{}.png","data/Flickr2K_HR/{}.png"], "Path of the train image folder.")
flags.DEFINE_string("valid_image_folder_path", "data/DIV2K_valid_HR/{}.png", "Path of the valid image folder.")
flags.DEFINE_list("test_image_folder_path", ["data/McM/*.png","data/kodak/*.png","data/B100/*.png","data/urban100/*.png"], "Path of the test image folder.")
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
flags.DEFINE_string("output_folder_path", "results", "Path to directory to output files.")
flags.DEFINE_string("optimizer", "adam", "optimizer of model")
flags.DEFINE_string("initializer", "he_normal", "initializer of model")
flags.DEFINE_integer("scale_factor", 2, "Scale factor 2, 3 or 4.")
flags.DEFINE_float("noise", 0.0784, "standard deviation of the Gaussian noise added to the images.")
flags.DEFINE_string("loss_function", "mean_absolute_error", "loss function of training.")
flags.DEFINE_bool("transfer", False, "transfer parameters from pretrained model")
flags.DEFINE_string("model_folder_path", "models/jdmsr+_model.h5", "Path of the trained model folder.")
FLAGS = flags.FLAGS

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
    transfer = FLAGS.transfer
    model_folder_path = FLAGS.model_folder_path

    shutil.rmtree(output_folder_path, ignore_errors=True)
    os.makedirs(output_folder_path)
    print("Recreating the output folder at {} ...".format(output_folder_path))

    print("Initiating the model ...")
    if max_noise > 0:
        add_noise = True
    else:
        add_noise = False
    
    model = get_model(optimizer, initializer, loss_function, filters, layers, scale_factor, add_noise)
    if transfer:
        old_model = load_model(model_folder_path)
        for i in range(738):
            if i < 2:
                model.layers[i].set_weights(old_model.layers[i].get_weights())
            elif i > 4:
                model.layers[i].set_weights(old_model.layers[i-2].get_weights())
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
