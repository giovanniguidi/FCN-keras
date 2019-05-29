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
from keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
#import keras.backend as K
#from keras.optimizers import Adam

from base.base_trainer import BaseTrain


class TrainerFCN(BaseTrain):
    
    def __init__(self, config, model, train_generator, val_generator):
        """
        Constructor
        """
        super().__init__(config, model, train_generator, val_generator)
        self.epochs = self.config['train']['num_epochs']
        #self.steps_per_epoch = self.config['train']['steps_per_epoch']        
        self.graph_path = self.config['network']['graph_path']
        self.weights_path = self.config['train']['output']['output_weights']
        self.callbacks_list = self.callbacks()

    def callbacks(self):
        """Create the callback list

        Returns
        -------
        callbacks : list
            callback list
        """
        
        callbacks = []

        #early stopping
        if self.config['callbacks']['early_stopping']['enabled'] == True:
            monitor = self.config['callbacks']['early_stopping']['monitor']
            patience = self.config['callbacks']['early_stopping']['patience']        
            callbacks.append(EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto'))

        #tensorboard
        if self.config['callbacks']['tensorboard']['enabled'] == True:
            log_dir = self.config['callbacks']['tensorboard']['log_dir']
            callbacks.append(TensorBoard(log_dir=log_dir))
            
        #best checkpoint
        if self.config['callbacks']['model_best_checkpoint']['enabled'] == True:
            monitor = self.config['callbacks']['model_best_checkpoint']['monitor']
            filepath = self.config['callbacks']['model_best_checkpoint']['out_file']
            callbacks.append(ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True, 
                                             save_weights_only=True, mode='min'))

        #last checkpoint
        if self.config['callbacks']['model_last_checkpoint']['enabled'] == True:
            filepath = self.config['callbacks']['model_last_checkpoint']['out_file']
            callbacks.append(ModelCheckpoint(filepath, verbose=1, save_best_only=False, 
                            save_weights_only=True))

        #reduce lr on plateau
        if self.config['callbacks']['reduce_lr_on_plateau']['enabled'] == True:
            monitor = self.config['callbacks']['reduce_lr_on_plateau']['monitor']
            factor = self.config['callbacks']['reduce_lr_on_plateau']['factor']
            patience = self.config['callbacks']['reduce_lr_on_plateau']['patience']
            callbacks.append(ReduceLROnPlateau(monitor = monitor, factor = factor, patience = patience))
            
        return callbacks

                             
    def train(self):
        """
        Train a model
        """

        #initialize weights
        if self.config['train']['weights_initialization']['use_pretrained_weights']:
            snapshot_file = self.config['train']['weights_initialization']['restore_from']
            
            print('Restoring weights from', snapshot_file)
            self.model.load_weights(snapshot_file)
        
        else:
            print('Saving graph in ', self.graph_path)
            self.save_graph(self.model, self.graph_path)
        
        #fit 
        use_multiprocessing = self.config['train']['use_multiprocessing']
        num_workers = self.config['train']['num_workers']
        
        self.model.fit_generator(generator=self.train_generator, validation_data=self.val_generator, 
                                 epochs=self.epochs, verbose=1, max_queue_size=10, 
                                 workers=num_workers, use_multiprocessing=use_multiprocessing, shuffle=False, 
                                 callbacks=self.callbacks_list)
        
        #save weights
        print("Saving weights in", self.weights_path)
        self.model.save_weights(self.weights_path)
    
    
    def save_graph(self, model, graph_path):
        
        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)
    
