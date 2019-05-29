import os
import numpy as np
import cv2
import io
import json
#import keras
import string

from base.base_data_generator import BaseDataGenerator
from data_generators.data_augmentation import data_aug_functions
from preprocessing.preproc_functions import read_image, normalize_0_mean_1_variance, normalize_0_1, read_annotation, convert_annotation_one_hot
from keras.applications.vgg16 import preprocess_input

class DataGenerator(BaseDataGenerator):
    def __init__(self, config, dataset, shuffle=True, use_data_augmentation=False):
        super().__init__(config, shuffle, use_data_augmentation) 
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.indices = np.arange(self.dataset_len)
        self.num_classes = self.config['network']['num_classes']
        self.on_epoch_end()
        
    def __len__(self):
        
        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):
        
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        dataset_temp = [self.dataset[k] for k in indices]
        
        # Generate data
        X, y = self.data_generation(dataset_temp)

        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch 
        """
        
        if self.shuffle == True:
            np.random.shuffle(self.indices)

        
    def data_generation(self, dataset_temp):        
        
        batch_x = []
        batch_y = []

        for elem in dataset_temp:            
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']
            convert_to_grayscale = self.config['image']['convert_to_grayscale']
                                     
            image = read_image(self.dataset_folder, elem['filename'], y_size, x_size, black_white = False)
            annotation = read_annotation(self.dataset_folder, elem['annotation'], y_size, x_size)
            
#            print(annotation.shape)
            
            if self.use_data_aug:
                #print('data aug')
                image, annotation = data_aug_functions(image, annotation, self.config)
            
#            annotation_one_hot = annotation
            annotation_one_hot = convert_annotation_one_hot(annotation, y_size, x_size, num_classes = self.num_classes)
#            image = normalize_0_1(image)                   
#            image = normalize_0_mean_1_variance(image)                   
            image = preprocess_input(image, mode='tf')  

            #print(image.shape)
            batch_x.append(image)
            batch_y.append(annotation_one_hot)
        
        batch_x = np.asarray(batch_x, dtype = np.float32)
        batch_y = np.asarray(batch_y, dtype = np.float32)
        
        return batch_x, batch_y
    
    
    def get_full_dataset(self):

        dataset_images = []
        dataset_labels = []
        
        for elem in self.dataset:
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']
            convert_to_grayscale = self.config['image']['convert_to_grayscale']

            image = read_image(self.dataset_folder, elem['filename'], y_size, x_size, black_white = False)            
            annotation = read_annotation(self.dataset_folder, elem['annotation'], y_size, x_size)
            
            annotation = read_annotation(self.dataset_folder, elem['annotation'], y_size, x_size)
            annotation_one_hot = convert_annotation_one_hot(annotation, y_size, x_size, num_classes = self.num_classes)
            
#            image = normalize_0_mean_1_variance(image)                   

            image = preprocess_input(image, mode='tf')
            
            dataset_images.append(image)
            dataset_labels.append(annotation_one_hot)
            
        dataset_images = np.asarray(dataset_images)
        dataset_labels = np.asarray(dataset_labels)
        
        return dataset_images, dataset_labels
    
