import os
import numpy as np
import cv2
import random
import io
from PIL import Image, ImageEnhance 

def data_aug_functions(img, annotation, config):
    """Function to augment data

    Parameters
    ------
    config: dict
        configuration file
        
    Returns
    -------
    img: numpy array
        augmented image
    """

    #print('data aug')
    
    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    num_channels = config['image']['image_size']['num_channels'] 
    
    rotation_range = config['data_aug']['rotation_range']
    x_shift_range = config['data_aug']['x_shift_range']
    y_shift_range = config['data_aug']['y_shift_range']
    zoom_range = config['data_aug']['zoom_range']
    horizontal_flip = config['data_aug']['horizontal_flip']
    vertical_flip = config['data_aug']['vertical_flip']
    shear_range = config['data_aug']['shear_range'] 
    brighntess_range = config['data_aug']['brightness_range'] 
    saturation_range = config['data_aug']['saturation_range'] 
    
    img_aug = img
    ann_aug = annotation

    img_aug, ann_aug = flip(img_aug, ann_aug, horizontal_flip, vertical_flip, y_size, x_size, num_channels)
    img_aug, ann_aug = translation(img_aug, ann_aug, y_shift_range, x_shift_range, y_size, x_size, num_channels)
    
    img_aug = random_brightness(img_aug, brighntess_range)
    img_aug = random_saturation(img_aug, saturation_range)

    
#      img_aug, ann_aug = rotation(img_aug, ann_aug, rotation_range, y_size, x_size, num_channels)
#    img_aug, ann_aug = zoom(img_aug, ann_aug, zoom_range, y_size, x_size, num_channels)

#    img_aug = shear(img_aug, shear_range, y_size, x_size, num_channels)

    return img_aug, ann_aug

def random_brightness(img, brighntess_range):

    brightness = 1. - np.random.uniform(-brighntess_range, brighntess_range)
    img_aug = Image.fromarray(img)
    
    enhancer = ImageEnhance.Brightness(img_aug)
    img_aug = enhancer.enhance(brightness)

    img_aug = np.array(img_aug)

    return img_aug

def random_saturation(img, saturation_range):

    saturation = 1. - np.random.uniform(-saturation_range, saturation_range)
    img_aug = Image.fromarray(img)
    
    enhancer = ImageEnhance.Color(img_aug)
    img_aug = enhancer.enhance(saturation)
    img_aug = np.array(img_aug)

    return img_aug


def translation(img, ann, y_shift_range, x_shift_range, y_size, x_size, num_channels):

    #print('translating')
    y_shift = np.random.uniform(-y_shift_range, y_shift_range) * y_size
    x_shift = np.random.uniform(-x_shift_range, x_shift_range) * x_size

    img = Image.fromarray(img)
    ann = Image.fromarray(ann)
    
#    a = 1
#    b = 0
#    c = x_shift #left/right (i.e. 5/-5)
#    d = 0
#    e = 1
#    f = y_shift #up/down (i.e. 5/-5)
    
    img_translated = img.transform(img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))   
    img_translated = np.array(img_translated)

    ann_translated = ann.transform(ann.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))   
    ann_translated = np.array(ann_translated)
    
#    M = np.float32([[1,0, x_shift],[0,1, y_shift]])
#    img_translated = cv2.warpAffine(img, M, (x_size, y_size))
#    ann_translated = cv2.warpAffine(ann, M, (x_size, y_size))

#    ann_translated = ann

    return img_translated, ann_translated



def flip(img, ann, horizontal_flip, vertical_flip, y_size, x_size, num_channels):
    """Function to flip images horizontally and vertically with prob = 0.5 

    Parameters
    ------
    img: numpy array
        configuration file
    horizontal_flip: bool
        configuration file
    vertical_flip: bool
        configuration file
    y_size: int
        image height
    x_size: int
        image width
    num_channels: int
        number of channels
        
    Returns
    -------
    img: numpy array
        flipped image
    """

    img_flipped = img
    ann_flipped = ann
    
    if vertical_flip:
        if np.random.randint(2) == 0:
#            print('flipping')
            img_flipped = cv2.flip(img_flipped, 1)
            ann_flipped = cv2.flip(ann_flipped, 1)
    if horizontal_flip:
        if np.random.randint(2) == 0:
            img_flipped = cv2.flip(img_flipped, 0)
            ann_flipped = cv2.flip(ann_flipped, 0)

    return img_flipped, ann_flipped


def translation_orig(img, ann, width_shift_range, height_shift_range, y_size, x_size, num_channels):
    """Function to translate images randomly between [-width_shift_range, width_shift_range]
       and [-width_shift_range, width_shift_range]

    Parameters
    ------
    img: numpy array
        configuration file
    width_shift_range: float
        maximum x shift range
    height_shift_range: float
        maximum y shift range
    y_size: int
        image height
    x_size: int
        image width
    num_channels: int
        number of channels
        
    Returns
    -------
    img: numpy array
        translated image
    """

    #print('translating')
    y_shift = np.random.uniform(-height_shift_range, height_shift_range) * y_size
    x_shift = np.random.uniform(-width_shift_range, width_shift_range) * x_size

    M = np.float32([[1,0, x_shift],[0,1, y_shift]])
    img_translated = cv2.warpAffine(img, M, (x_size, y_size))
    ann_translated = cv2.warpAffine(ann, M, (x_size, y_size))

    return img_translated, ann_translated


def rotation(img, ann, rotation_range, y_size, x_size, num_channels):
    """Function to rotate images randomly between [-rotation_range, rotation_range]

    Parameters
    ------
    img: numpy array
        configuration file
    rotation_range: float
        maximum rotation range
    y_size: int
        image height
    x_size: int
        image width
    num_channels: int
        number of channels
        
    Returns
    -------
    img: numpy array
        rotated image
    """
    
    angle = np.random.uniform(-rotation_range, rotation_range)

    M = cv2.getRotationMatrix2D((x_size/2, y_size/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (x_size, y_size))
    ann_rot = cv2.warpAffine(ann, M, (x_size, y_size))

    img_rot = np.reshape(img_rot, (y_size, x_size, num_channels))
#    ann_rot = np.reshape(ann, (y_size, x_size, num_channels))

    return img_rot, ann_rot



def zoom(img, ann, zoom_range, y_size, x_size, num_channels):
    """Function to zoom images randomly between [-zoom_range, zoom_range]

    Parameters
    ------
    img: numpy array
        configuration file
    zoom_range: float
        maximum zoom range
    y_size: int
        image height
    x_size: int
        image width
    num_channels: int
        number of channels
        
    Returns
    -------
    img: numpy array
        zoomed image
    """
    
    zoom = np.random.uniform(zoom_range[0], zoom_range[1]) 

    p1 = [5, 5] 
    p2 = [20, 5]
    p3 = [5, 20]

    pts1 = np.float32([p1, p2, p3])
    pts2 = np.float32([[x * zoom for x in p1], 
                       [x * zoom for x in p2], 
                       [x * zoom for x in p3] ])

    M = cv2.getAffineTransform(pts1,pts2) 
    zoomed_image = cv2.warpAffine(img, M, (x_size, y_size))
    zoomed_ann = cv2.warpAffine(ann, M, (x_size, y_size))

    return zoomed_image, zoomed_ann


def shear(img, shear_range, y_size, x_size, num_channels):
    """Function to shear images randomly between [-shear_range, shear_range]

    Parameters
    ------
    img: numpy array
        configuration file
    shear_range: float
        maximum shear range
    y_size: int
        image height
    x_size: int
        image width
    num_channels: int
        number of channels
        
    Returns
    -------
    img: numpy array
        sheared image
    """
    
    pts1 = np.float32([[5,5],[20,5],[5,20]])
    pt1 = 5 + shear_range*np.random.uniform() - shear_range/2
    pt2 = 20 + shear_range*np.random.uniform() - shear_range/2
    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    M = cv2.getAffineTransform(pts1, pts2) 
    sheared_range = cv2.warpAffine(img, M, (x_size, y_size))

    return sheared_range
