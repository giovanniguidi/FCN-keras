# Fully Convolutional Network for Semantic Segmentation

This project is an implementation of the Fully Convolutional Network for Semantic Segmentation model (Long et al. 2015, https://arxiv.org/pdf/1411.4038.pdf) trained on VOC2012, although this implementation can be easily retrained on other datasets. The task this algorithm wants to solve is "semantic segmentation", i.e. given a picture assign each pixel a "semantic" label, such as tree, street, sky, car. 

![picture alt](https://github.com/giovanniguidi/FCN-keras/tree/master/figures/semantic_segmentation.jpg "")

The model is made by two parts, the "encoder" which is a standard convolutional network (VGG16 in this case following the paper), and the decoder which upsamples the result of the encoder to the full resolution of the original image using transposed convolutions. Skips between the encoder and decoder ensure that the spatial information from early layers of the encoder is passed to the decoder, increasing the localization accuracy of the model. 

![picture alt](https://github.com/giovanniguidi/FCN-keras/tree/master/figures/FCN_1.jpg "")

In the paper the authors use for the encoder weights pretrained on ImageNet, and tested three different decoders with increasing strides (32x, 16x, 8x) and corresponding better values of the metrics.


## Depencencies

Install the libraries using:
```
pip install -r requirements.txt 
```

## Data

The dataset used for this project is the Pascal Visual Object Classes Challenge 2012 VOC2012 (use the kaggle mirror if the official website is unavailable, https://www.kaggle.com/huanghanchina/pascal-voc-2012/downloads/pascal-voc-2012.zip/1).

This dataset contains ~2500 images belonging to 20 classes (person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor). 

You need to untar this file into "./datasets" folder to use this project without changing the config.yml file. 

This is an example of the images in the dataset:

![picture alt](https://github.com/giovanniguidi/FCN-keras/tree/master/test_images/008_005541.jpg "")

Modifying the data_generation function in data_generator.py the model can be easily trained on other data.


## Project structure

The project has this structure:

- base: base classes for data_generator, model, trainer and predictor 

- callbacks: custom callbacks 

- configs: configuration file

- data_generators: data generator class and data augmentation functions

- datasets: folder containing the dataset and the labels

- experiments: contains snapshots, that can be used for restoring the training 

- figures: plots and figures

- losses: custom losses

- models: neural network model

- notebooks: notebooks for testing 

- predictors: predictor class 

- preprocessing: preprocessing functions (reading and normalizing the image)

- snapshots: graph and weights of the trained model

- tensorboard: tensorboard logs

- test_images: images from the dataset that can be used for testing 

- trainers: trainer classes

- utils: various utilities, including the one to generate the labels.json


## Input

The input json is created by the script in utils/create_labels.py and follows this structure:

```
dataset['train'], ['val'], ['test']
```

Each split gives a list of dictionary: {'filename': FILENAME, 'annotation': ANNOTATION}.


## Weights

The graph and trained weights can be found at:

https://drive.google.com/open?id=1JXfM5X0aihv2d_4WN8_bIvzrfhB0Me5k


If you want to use these weights be sure that you keep the original train/val/test split (ie. use the original labels.json in "datasets"), otherwise you may mix the train and test set and you results will be unreliable.


## Train

To train a model run:

```
python main.py -c configs/config.yml --train
```

If you set "weights_initialization" in config.yml you can use a pretrained model to inizialize the weights, usually for restoring the training after an interruption.  

During training the best and last snapshots can be stored if you set those options in "callbacks" in config.yml.


## Inference 

To predict on the full test set run: 

```
python main.py -c configs/config.yml --predict_on_test
```

(you need labels.json in "datasets" folder).


In "./test_images/" there are some images that can be used for testing the model. 

To predict on a single image you can run:

```
python main.py -c configs/config.yml --predict --filename test_images/test_images/2010_004856.jpg
```


## Results

Here is an example of prediction:

![picture alt](https://github.com/giovanniguidi/FCN-keras/tree/master/figures/pred_3.png "")

Check "inference.ipynb" in notebooks for a visual assessment of the prediction.

On the test set we get this metrics (see https://arxiv.org/pdf/1411.4038.pdf for the definition):

```
pixel accuracy: 0.81
mean accuracy: 0.35
mean IoU: 0.27
freq weighted mean IoU: 0.69
````

## Train on other data

To use this implementation on other datasets you need first to create the labels.json file by running the script in utils/create_labels.py. Then you should modify the functions in data_generator.py to read your images and annotations. 

The output of data_generator.py are two numpy arrays, one containing the images (reshaped and normalized) and the other the annotations with shape (batch_size, y_size, x_size, one_hot_encoded_classes).
 
When training you can randomly initialize the encoder instead of using pretrained weights on ImageNet, by setting "train_from_scratch: true" in the config.yml file.

## Technical details

Images are normalized between -1 and 1.

Data can be augmented by flipping, translating, and changing random_brightness and random_saturation.


## To do

- [x] 


## References


\[1\] [Fully Convolutional Network for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
