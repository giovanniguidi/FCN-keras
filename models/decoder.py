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
from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, Add, ZeroPadding2D, Cropping2D
#from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam

from base.base_model import BaseModel
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json


def decoder_graph_8x(pool_3, pool_4, encoder_out, num_classes):

    # Unpool to 16x
    score_2 = Conv2DTranspose(num_classes, 4, strides=(2, 2), padding='valid')(encoder_out)
    score_pool_4 = Conv2D(num_classes, 1, padding='valid', use_bias=True)(pool_4)
    score_pool_4 = Cropping2D(cropping=5)(score_pool_4)
    score_16x_upsampled = Add()([score_2, score_pool_4])
        
    # Unpool to 8x
    score_4 = Conv2DTranspose(num_classes, 4, strides=(2, 2), padding='valid')(score_16x_upsampled)
    score_pool_3 = Conv2D(num_classes, 1, padding='valid', use_bias=True)(pool_3)
    score_4 = ZeroPadding2D(padding=((1,0), (1, 0)))(score_4)
    score_pool_3 = Cropping2D(cropping=9)(score_pool_3)
    score_8x_upsampled = Add()([score_4, score_pool_3])

    # Unpool to image shape
    upsample = Conv2DTranspose(num_classes, 16, strides=(8, 8), padding='same')(score_8x_upsampled)        
    upsample = Cropping2D(cropping=28)(upsample)
    
    output_graph = Activation('softmax')(upsample)

#    print(output_graph.shape)
    
    return output_graph


def decoder_graph_16x(pool_4, encoder_out, num_classes):

    # Unpool to 16x
    score_2 = Conv2DTranspose(num_classes, 4, strides=(2, 2), padding='same')(encoder_out)
    score_pool_4 = Conv2D(num_classes, 1, padding='same', use_bias=True)(pool_4)    
    score_pool_4 = Cropping2D(cropping=6)(score_pool_4)
    score_16x_upsampled = Add()([score_2, score_pool_4])
    
    # Unpool to image shape
    upsample = Conv2DTranspose(num_classes, 32, strides=(16, 16), padding='same')(score_16x_upsampled)        
    output_graph = Activation('softmax')(upsample)

    return output_graph


def decoder_graph_32x(encoder_out, num_classes):

    # Unpool to image shape
    upsample = Conv2DTranspose(num_classes, 64, strides=(32, 32), padding='same')(encoder_out)        
#    upsample = Conv2DTranspose(num_classes, 224, strides=(224, 224), padding='same')(encoder_out)        

    output_graph = Activation('softmax')(upsample)

    return output_graph