#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wenzhu, Xing
wenzhu.xing@tuni.fi
"""
"""Models for 'Joint Demosaicking, Denoising and Super-Resolution'."""

from keras.layers import Conv2D, Multiply, Add, GlobalAveragePooling2D, Reshape, Dense, Lambda

def CA(input_tensor, filters, reducer=16):
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, filters))(x)
    x = Dense(filters//reducer,  activation='relu', kernel_initializer='he_normal', use_bias=False)(x)
    x = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)
    x = Multiply()([x, input_tensor])
    return x

def RCAB(input_tensor, filters, scale=0.1):
    """
    Residual Channel Attention Block
    style: 2Conv.->CA-+->
               |______|
    """
    kernel_size = (3, 3)
    x = Conv2D(filters, kernel_size, padding='same', data_format='channels_last',
               activation='relu')(input_tensor)
    x = Conv2D(filters, kernel_size, padding='same', data_format='channels_last')(x)
    x = CA(x, filters)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])
    return x

def RG(input_tensor, filters, n_RCAB=20):
    """
    Residual Group
    style: 20 RCAB
    The core module of paper: (RCAN)
    """
    x = input_tensor
    for i in range(n_RCAB):
        x = RCAB(x, filters)
    kernel_size = (3, 3)
    rg = Conv2D(filters, kernel_size, padding='same', data_format='channels_last')(x)
    rg = Add()([rg, input_tensor])
    return rg
