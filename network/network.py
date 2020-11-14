#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Wenzhu, Xing
wenzhu.xing@tuni.fi
"""
"""network of 'Joint Demosaicking, Denoising and Super-Resolution'."""

import keras
from keras.layers import Input, Conv2D, Conv2DTranspose, Add
from keras.models import Model
from network.model import RG

import sys
sys.setrecursionlimit(10000)

def get_model(optimizer, initializer, loss_function, filters, depth, scale_factor):

    mosaick = Input(shape=(None, None, 1))
    estimated_noise = Input(shape=(None, None, 1)) 

    # Set the input size and filter size
    kernel_size = (3, 3)
    
    # Down-sample
    pack_mosaick = Conv2D(4, (2, 2), strides=(2, 2), padding='same',
                          data_format='channels_last',
                          kernel_initializer=initializer)(mosaick)
    
    # Add estimated noise vector
    mosaick_noise = keras.layers.concatenate([pack_mosaick, estimated_noise], axis=3)

    # Color extraction
    first_conv = Conv2D(filters*4, kernel_size, padding='same',
                        data_format='channels_last', activation='relu',
                        kernel_initializer=initializer)(mosaick_noise)
    
    # Up-sample (deconv.)
    first_conv = Conv2DTranspose(filters, kernel_size, strides=(2, 2), padding='same',
                                 data_format='channels_last', activation='relu',
                                 kernel_initializer=initializer)(first_conv)

    # Feature extraction
    for layer_id in range(depth):
        if layer_id == 0:
            features = RG(first_conv, filters)
        else:
            features = RG(features, filters)
            
    # long skip connection
    final_conv = Conv2D(filters, kernel_size, padding='same',
                        data_format='channels_last', activation='relu',
                        kernel_initializer=initializer)(features)
    
    final_conv = Add()([first_conv, final_conv])
    
    # Reconstruction
    upsample = Conv2DTranspose(filters, kernel_size, strides=(scale_factor, scale_factor), padding='same',
                               data_format='channels_last', activation='relu',
                               kernel_initializer=initializer)(final_conv)
    
    output = Conv2D(3, kernel_size, padding='same',
                    data_format='channels_last',
                    kernel_initializer=initializer)(upsample)
    
    model = Model([mosaick, estimated_noise], output)
#        model = Model(mosaick, output)

    # Compile the model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=None)

    return model
