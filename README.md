# Fully Convolutional Network for Semantic Segmentation

This project is an implementation of the Fully Convolutional Network for Semantic Segmentation (Long et al. 2015, https://arxiv.org/pdf/1411.4038.pdf). The task this algorithm wants to solve is semantic segmentation, namely given a picture we want to assign each label a "semantic" meaning, such as tree, street, sky, car.   HERE IMAGE


![picture alt](https://github.com/giovanniguidi/Seq-2-Seq-OCR/blob/master/figures/seq2seq.png "")

The model in the paper is made by two parts, the "encoder" which is a standard convolutional network (in this case VGG16) and the decoder which is responsible to upsample the result of the decoder to the full resolution of the original image. Skips between teh encoder and decoder ensure that the spatial information from early layers of the encoder is preserved, increasing the localization accuracy of the model. 

In the paper they propose 3 types of decoder which they called 8x, 16x, 32x according to the stride of the network.

The most accurate one is 8x, 

They use pretrained weights on ImageNet for VGG16 network.


## Depencencies

Install the libraries using:
```
pip install -r requirements.txt 
```

## Data


The dataset is from the Pascal Visual Object Classes Challenge 2011 (VOC2011)

The dataset contains around 2000 images belonging to 20 classes (person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, dining table, potted plant, sofa, tv/monitor). 

The dataset can be downloaded at:

http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar

If you want to use this project you need to untar it into "./datasets" folder. 

This is an example of images in the dataset:

![picture alt](https://github.com/giovanniguidi/Seq-2-Seq-OCR/blob/master/test_images/b01-049-01-00.png "")

Modifying the data generator the model can be easily trained on other data


## Project structure

The project has this structure:

- base: base classes for data_generator, model, trainer and predictor 

- callbacks: custom callbacks (unused)

- configs: configuration file

- data_generators: data generator class and data augmentation functions

- datasets: folder containing the dataset and the labels

- experiments: contains snapshots, that can be used for restoring the training 

- figures: plots and figures

- models: neural network model

- notebooks: notebooks for testing 

- predictors: predictor class 

- preprocessing: preprocessing functions (reading and normalizing the image)

- snapshots: graph and weights of the trained model

- tensorboard: tensorboard logs

- test_images: images from the dataset that can be used for testing 

- trainers: trainer classes

- utils: various utilities, including the one to generate the labels


## Input

The input json can be created from utils/create_labels.py and follows this structure:

```
dataset['train'], ['val'], ['test']
```

Each split gives a list of dictionary: {'filename': FILENAME, 'label': LABEL}.


## Weights

The graph and trained weights can be found at:

https://drive.google.com/open?id=1JXfM5X0aihv2d_4WN8_bIvzrfhB0Me5k


If you want to use these weights be sure that you keep the original dataset split (use the original labels.json in "datasets"), otherwise you may mix the train and test set and you results will be unreliable.


## Train

To train a model run:

```
python main.py -c configs/config.yml --train
```

If you set "weights_initialization" in config.yml you can use a pretrained model to inizialize the weights. 

During training the best and last snapshots can be stored if you set those options in "callbacks" in config.yml.


## Inference 

To predict on the full test set run: 

```
python main.py -c configs/config.yml --predict_on_test
```

(you need a file labels.json in "dataset").


In "./test_images/" there are some images that can be used for testing the model. 

To predict on a single image you can run:

```
python main.py -c configs/config.yml --predict --filename test_images/test_images/f07-036-02-02.png
```


## Performance

On the test set we get this performance (character error rate and word error rate):

```
CER:  12.62 %
WER:  26.65 %
```

########### dropout 0.5:   

----- train ----- 

pixel accuracy: 0.86
mean accuracy: 0.49
mean IoU: 0.36
freq_weighted_mean_IoU: 0.77


----- test ------

pixel accuracy: 0.75
mean accuracy: 0.21
mean IoU: 0.14
freq_weighted_mean_IoU: 0.62

################## baseline


pixel acc.    mean acc.   mean IU     f.w. IU

90.3          75.9         62.7        83.2


## Technical details

Images are normalized between -1 and 1

Data augmentation is used



## To do

- [x] -----


## References


\[1\] [Fully Convolutional Network for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
