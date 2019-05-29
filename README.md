### Performance

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


########### pretraining:


----- train ----- 




----- test ------






########### data aug:  

----- train ----- 




----- test ------



########### dropout + data aug:

----- train ----- 


----- test ------


################## baseline


pixel acc.    mean acc.   mean IU     f.w. IU

90.3          75.9         62.7        83.2



# FCN keras


## Depencencies



## Data


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

## Dataset

http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar

## Input

The input json can be created from utils/create_labels.py and follows this structure:

```
dataset['train'], ['val'], ['test']
```

Each split gives a list of dictionary: {'filename': FILENAME, 'label': LABEL}.


## Weights


## Train


## Inference 

To predict on the full test set run: 

```
python main.py -c configs/config.yml --predict_on_test
```

(you need a file labels.json in "dataset").


In "./test_images/" there are some images that can be used for testing the model. 

To predict on a single image you can run:

```
python main.py -c configs/config.yml --predict --filename test_images/test_images/
```


## Performance


## To do



## References
