import os
import numpy as np
import cv2
import random
#import datetime
import io
import json
import keras
import string


from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from keras.callbacks import EarlyStopping
#import keras.backend as K
from keras.optimizers import Adam
from keras.models import model_from_json
from preprocessing.preproc_functions import read_image, normalize_0_mean_1_variance

from base.base_predictor import BasePredictor

class PredictorFCN(BasePredictor):
    
    def __init__(self, config):
        """
        Constructor
        """
        super().__init__(config)
        self.graph_path = self.config['network']['graph_path']
        self.weights_path = self.config['predict']['weights_file']

        self.batch_size = self.config['predict']['batch_size']
        self.model = self.load_model()
    

    def load_model(self):
        
        json_file = open(self.graph_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(self.weights_path)
        
        return model
        
    def predict(self, images):

        batch_size = self.batch_size
        
        n_images = images.shape[0]
        y_size = images.shape[1]
        x_size = images.shape[2]

        n_batches = (n_images + batch_size - 1) // batch_size

        output_list = []

        for i in range(n_batches):
    #    for i in range(1):

            batch_in, batch_out = (batch_size)* i, (batch_size)* i + batch_size

            if batch_out >= n_images:
                batch_out = n_images

            input_batch = images[batch_in:batch_out, :, :, :]
            batch_dim = batch_out - batch_in
            output_batch = self.model.predict(input_batch, batch_size = batch_dim)

            output_list.append(output_batch)

            #flatten list
        flattened_list = [item for sublist in output_list for item in sublist]
        pred_out = np.asarray(flattened_list)
        
        return pred_out

