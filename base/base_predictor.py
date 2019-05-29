import os
import numpy as np
#import cv2
#import string

class BasePredictor(object):
    """
    Base class for prediction

    Attributes
    ----------
    config : dict
        configuration file 

    Methods
    -------
    predict(images)
        returns the prediction given an array of images
    load_model(graph_path, weights_path)
        create the graph from a json and load the weights 
    """
    
    def __init__(self, config):
        """
        Base constructor
        """
        self.config = config
        
    def predict(self, images):
        """returns the prediction given an array of images

        Parameters
        ------
        images: np.array(batch_size, y_size, x_size, num_channels) 
            array containing the images
            
        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError
        
    def load_model(self, graph_path, weights_path):
        """create the graph from a json and load the weights 

        Parameters
        ------
        graph_path: str
            path to the json with graph 
        weights_path: str
            path to the weights (.h5 format)
            
        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError