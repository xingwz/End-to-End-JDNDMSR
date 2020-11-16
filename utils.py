#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wenzhu, Xing
wenzhu.xing@tuni.fi
"""

from keras.callbacks import Callback
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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
