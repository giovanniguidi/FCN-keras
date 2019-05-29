import os
import numpy as np
import cv2
import random
import datetime
import io
import json
import keras
import string


from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, Add, ZeroPadding2D
#from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam

from base.base_model import BaseModel
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json


def encoder_graph(y_size, x_size, num_channels, num_classes):

    input_graph = Input(shape=(y_size, x_size, num_channels), name='input_image')

    graph = ZeroPadding2D(padding=100)(input_graph)

    #block_1
    graph = Conv2D(64, 3, padding='same', activation='relu', use_bias=True, name='block1_conv1')(graph)
    graph = Conv2D(64, 3, padding='same', activation='relu', use_bias=True, name='block1_conv2')(graph)
    graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block1_pool')(graph)
    
    #block_2
    graph = Conv2D(128, 3, padding='same', activation='relu', use_bias=True, name='block2_conv1')(graph)
    graph = Conv2D(128, 3, padding='same', activation='relu', use_bias=True, name='block2_conv2')(graph)
    graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block2_pool')(graph)

    #block_3
    graph = Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv1')(graph)
    graph = Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv2')(graph)
    graph = Conv2D(256, 3, padding='same', activation='relu', use_bias=True, name='block3_conv3')(graph)
    graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block3_pool')(graph)
#    graph = Dropout(0.4)(graph)

    pool_3 = graph

#    print(pool_3.shape)

    #block_4
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv1')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv2')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block4_conv3')(graph)
    graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block4_pool')(graph)
#    graph = Dropout(0.4)(graph)
        
    pool_4 = graph

#    print(pool_4.shape)
    
    #block_5
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv1')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv2')(graph)
    graph = Conv2D(512, 3, padding='same', activation='relu', use_bias=True, name='block5_conv3')(graph)
    graph = MaxPooling2D(pool_size=(2, 2), padding='valid', name='block5_pool')(graph)
    
    #fc_6
    graph = Conv2D(4096, 7, padding='valid', activation='relu', use_bias=True, name='fc_1')(graph)
    graph = Dropout(0.5)(graph)
    
    #fc_7
    graph = Conv2D(4096, 1, padding='valid', activation='relu', use_bias=True, name='fc_2')(graph)
    graph = Dropout(0.5)(graph)

    encoder_graph = Conv2D(num_classes, 1, padding='valid', activation='relu', use_bias=True, name='encoder_graph')(graph)
    
#    print(encoder_graph.shape)
    
    return input_graph, pool_3, pool_4, encoder_graph


def encoder_graph_vgg16(num_classes):

    vgg16 = VGG16(weights='imagenet', include_top=True)

    input_graph = vgg16.input

    padding = ZeroPadding2D(padding=100)(input_graph)

    pool_1_vgg16 = vgg16.get_layer('block1_pool')
    pool_2_vgg16 = vgg16.get_layer('block2_pool')
    pool_3_vgg16 = vgg16.get_layer('block3_pool')
    pool_4_vgg16 = vgg16.get_layer('block4_pool')
    pool_5_vgg16 = vgg16.get_layer('block5_pool')
    fc_1_vgg16 = vgg16.get_layer('fc1')
    fc_2_vgg16 = vgg16.get_layer('fc2')

    #convolutionize fc_1 and fc_2
    fc_1 = convolutionize_dense_layer(fc_1_vgg16, (7, 7, 512, 4096))
    fc_2 = convolutionize_dense_layer(fc_2_vgg16, (1, 1, 4096, 4096))
    
    pool_1 = pool_1_vgg16(padding)
    pool_2 = pool_2_vgg16(pool_1)
    pool_3 = pool_3_vgg16(pool_2)
    pool_4 = pool_4_vgg16(pool_3)
    pool_5 = pool_5_vgg16(pool_4)
        
#    graph = Conv2D(4096, 7, padding='valid', activation='relu', use_bias=True, name='fc_1')(pool_5)
    
    print(pool_5.shape)
    
    graph = fc_1(pool_5)
    graph = Conv2D(4096, 1, padding='valid', activation='relu', use_bias=True)(graph)
       
#    graph = fc_1(pool_5)
#    graph = fc_2(graph)
        
#    print(graph.shape)
#    vgg16_out = vgg16.output
    #fc_6
#    encoder_graph = Conv2D(4096, 7, padding='same', activation='relu', use_bias=True, name='fc_6')(vgg16_out)
    #fc_7
#    graph = Conv2D(4096, 1, padding='same', activation='relu', use_bias=True, name='fc_7')(graph)

    encoder_graph = Conv2D(num_classes, 1, padding='same', activation='relu', use_bias=True, name='encoder_graph')(graph)

    print(encoder_graph.shape)
    
    return input_graph, pool_3, pool_4, encoder_graph

def convolutionize_dense_layer(layer, out_dim):
    
    W, b = layer.get_weights()
    W_reshaped = W.reshape(out_dim)
    
    conv_layer = Conv2D(out_dim[3], (out_dim[0], out_dim[1]), strides=(1,1), activation='relu', padding='valid',weights=[W_reshaped, b])
    
    return conv_layer
    
