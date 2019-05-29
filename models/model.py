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
from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Conv2DTranspose, Add
#from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam

from base.base_model import BaseModel
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import model_from_json
from losses.custom_losses import custom_categorical_crossentropy

from models.encoder import encoder_graph, encoder_graph_vgg16
from models.decoder import decoder_graph_8x, decoder_graph_16x, decoder_graph_32x


class ModelFCN(BaseModel):
    
    def __init__(self, config):
        """
        Constructor
        """
        super().__init__(config)
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.num_classes = self.config['network']['num_classes']
        self.use_pretrained_weights = self.config['train']['weights_initialization']['use_pretrained_weights']
        self.graph_path = self.config['network']['graph_path']
        self.decoder = self.config['network']['decoder']
        self.model = self.build_model()

    def build_model(self):
        
        model = self.build_graph()        
#        model.compile(optimizer = self.optimizer, loss = self.loss)
        model.compile(optimizer = self.optimizer, loss = custom_categorical_crossentropy())

#        model.summary()

        return model
    
        
    def build_graph(self):
        
        if self.use_pretrained_weights:
            json_file = open(self.graph_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()

            model = model_from_json(loaded_model_json)
            
        else:    
            input_graph, pool_3, pool_4, encoder_out = encoder_graph(self.y_size, self.x_size, self.num_channels, self.num_classes)

#            print(encoder_out.shape)
            
#            input_graph, pool_3, pool_4, encoder_out = encoder_graph_vgg16(self.y_size, self.x_size, self.num_channels, self.num_classes)

#            print(encoder_out.shape)

            if self.decoder == 'decoder_8x':
                decoder_out = decoder_graph_8x(pool_3, pool_4, encoder_out, self.num_classes)
                
            elif self.decoder == 'decoder_16x':
                decoder_out = decoder_graph_16x(pool_4, encoder_out, self.num_classes)

            elif self.decoder == 'decoder_32x':
                decoder_out = decoder_graph_32x(encoder_out, self.num_classes)

            else:
                raise Exception("Unknown decoder")
            
            model = Model(input_graph, decoder_out)
            
        return model

   