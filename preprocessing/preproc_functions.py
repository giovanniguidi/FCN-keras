import os
#import cv2
import numpy as np
from PIL import Image, ImageStat
from scipy.io import loadmat

def read_image(dataset_folder, filename, y_size, x_size, black_white = False):

    fpath = os.path.join(dataset_folder, filename)
    img = Image.open(fpath)
    
    img = img.resize((x_size, y_size), Image.ANTIALIAS)
    
    if black_white == True:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    img = np.array(img)
    
    
    return img 

def zoom_at(img, x, y, zoom):
    w, h = img.size
    zoom2 = zoom * 2
    
    factor = 1.
    
    img = img.crop(( (x - w)*factor / zoom2, y - h / zoom2, (x + w)*factor / zoom2, y + h / zoom2))

    return img

def convert_annotation_one_hot(annotation, y_size, x_size, num_classes):

    annotation_one_hot = np.zeros((y_size, x_size, num_classes))
    
    for i in range(y_size):
        for j in range(x_size):
            pix_value = annotation[i, j]
            annotation_one_hot[i, j, pix_value] = 1
        
    return annotation_one_hot 

def read_annotation(dataset_folder, filename, y_size, x_size):

    fpath = os.path.join(dataset_folder, filename)
    annotation = Image.open(fpath)
    
    annotation = annotation.resize((x_size, y_size))
    
#    annotation = annotation.convert('L')
 
    annotation = np.array(annotation)
    annotation[annotation == 255.] = 0.
    
    return annotation 


def normalize_0_mean_1_variance(img):

    img = Image.fromarray(img)

    stats = ImageStat.Stat(img)
    mean = stats.mean
    stddev = stats.stddev

    img = np.array(img)

    img = img - mean
    img = img / stddev
    
    return img


def normalize_0_1(img):

    img = img.astype('float')
    img /= 255. 

    return img

def zoom_image(img):

    x_shift = 20
    y_shift = 20
    
    img = img[x_shift:-x_shift, x_shift:-x_shift, :]
    
    
    return img