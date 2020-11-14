#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from keras.utils import Sequence

class DataSequence(Sequence):
    def __init__(self, whole_image_file_path_list, GT_image_file_path_list, image_height, image_width, batch_size, steps_per_epoch, scale_factor, max_noise):
        super(DataSequence, self).__init__()
        self.whole_image_file_path_list = whole_image_file_path_list
        self.GT_image_file_path_list = GT_image_file_path_list
        self.image_height, self.image_width = image_height, image_width
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.scale_factor = scale_factor
        self.max_noise = max_noise
        self.image_num_per_epoch = batch_size * steps_per_epoch
        
        self.current_image_file_path_list_generator = self._get_current_image_file_path_list_generator()
        self.current_image_file_path_list, self.current_gt_image_file_path_list = next(self.current_image_file_path_list_generator)
        
    def _get_current_image_file_path_list_generator(self):
        current_image_file_path_list = []
        current_gt_image_file_path_list = []
        whole_image_file_path_index_array = np.arange(len(self.whole_image_file_path_list))
        
        while True:
            np.random.shuffle(whole_image_file_path_index_array)
            for whole_image_file_path_index in whole_image_file_path_index_array:
                current_image_file_path_list.append(self.whole_image_file_path_list[whole_image_file_path_index])
                current_gt_image_file_path_list.append(self.GT_image_file_path_list[whole_image_file_path_index])
                if len(current_image_file_path_list) == self.image_num_per_epoch:
                    yield current_image_file_path_list, current_gt_image_file_path_list
                    current_image_file_path_list = []
                    current_gt_image_file_path_list = []
    
    def __len__(self):
        return self.steps_per_epoch
    
    def _data_aug(self, LR_image_content, clean_image_content, mode=0):
        if mode == 0:
            return LR_image_content, clean_image_content
        elif mode == 1:
            return np.flipud(LR_image_content), np.flipud(clean_image_content)
        elif mode == 2:
            return np.rot90(LR_image_content), np.rot90(clean_image_content)
        elif mode == 3:
            return np.flipud(np.rot90(LR_image_content)), np.flipud(np.rot90(clean_image_content))
        elif mode == 4:
            return np.rot90(LR_image_content, k=2), np.rot90(clean_image_content, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(LR_image_content, k=2)), np.flipud(np.rot90(clean_image_content, k=2))
        elif mode == 6:
            return np.rot90(LR_image_content, k=3), np.rot90(clean_image_content, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(LR_image_content, k=3)), np.flipud(np.rot90(clean_image_content, k=3))
    
    
    def _add_gaussian_noise(self, clean_image_content, sigma):
        corrupt_image_content = clean_image_content.copy()
        noise = np.random.normal(0, sigma, corrupt_image_content.shape)
        corrupt_image_content = corrupt_image_content.astype(np.float32) + noise.astype(np.float32)
        corrupt_image_content = np.clip(corrupt_image_content, 0, 255).astype(np.uint8)
        return corrupt_image_content
    
    
    def _bayer_mosaic(self, clean_image_content, pixel_order='rggb'):
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
    
    def __getitem__(self, index):
        if self.max_noise > 0:
            img_input_list, noise_input_list, output_list = [], [], []
        else:
            img_input_list, output_list = [], []
        image_file_path_list = self.current_image_file_path_list[index * self.batch_size:(index + 1) * self.batch_size]
        image_file_path_list_gt = self.current_gt_image_file_path_list[index * self.batch_size:(index + 1) * self.batch_size]
        for image_file_path, image_file_path_gt in zip(image_file_path_list, image_file_path_list_gt):
            # Read image
            LR_image_content = cv2.imread(image_file_path, 1)
            image_content = cv2.imread(image_file_path_gt, 1)
            LR_image_height, LR_image_width, _ = LR_image_content.shape
            assert LR_image_height >= self.image_height and LR_image_width >= self.image_width, "{} is problematic.".format(image_file_path)

            # Get the coordinates of the top left point
            LR_height_start_index = np.random.choice(LR_image_height - self.image_height + 1)
            LR_width_start_index = np.random.choice(LR_image_width - self.image_width + 1)
            height_start_index = LR_height_start_index * self.scale_factor
            width_start_index = LR_width_start_index * self.scale_factor

            # Crop the image
            LR_image_content = LR_image_content[LR_height_start_index:LR_height_start_index + self.image_height, LR_width_start_index:LR_width_start_index + self.image_width]
            clean_image_content = image_content[height_start_index:height_start_index + self.image_height * self.scale_factor, width_start_index:width_start_index + self.image_width * self.scale_factor]

            # Data aug
            LR_image_content, clean_image_content = self._data_aug(LR_image_content, clean_image_content, mode=np.random.randint(0,8))
            
            # Add the noise
            sigma = max(0, np.random.rand(1)*self.max_noise)[0]
            if sigma > 0:
                noise_image_content = self._add_gaussian_noise(LR_image_content, sigma*255)
                estimate_noise_input = sigma * np.ones((self.image_height // 2, self.image_width // 2, 1))
                noise_input_list.append(estimate_noise_input)
                # Bayer mosaick
                corrupt_image_content = self._bayer_mosaic(noise_image_content)
            else:
                corrupt_image_content = self._bayer_mosaic(LR_image_content)

            # Add the entries
            img_input_list.append(corrupt_image_content)
            output_list.append(clean_image_content)

        assert len(img_input_list) == self.batch_size
        if self.max_noise > 0:
            img_input_array, noise_input_array, output_array = np.array(img_input_list, dtype=np.float32) / 255, np.array(noise_input_list, dtype=np.float32), np.array(output_list, dtype=np.float32) / 255
            img_input_array = (img_input_array - 0.5) / 0.5
            output_array = (output_array - 0.5) / 0.5
            return [img_input_array, noise_input_array], output_array
        else:
            img_input_array, output_array = np.array(img_input_list, dtype=np.float32) / 255, np.array(output_list, dtype=np.float32) / 255
            img_input_array = (img_input_array - 0.5) / 0.5
            output_array = (output_array - 0.5) / 0.5
            return img_input_array, output_array
            

    def on_epoch_end(self):
        self.current_image_file_path_list, self.current_gt_image_file_path_list = next(self.current_image_file_path_list_generator)
