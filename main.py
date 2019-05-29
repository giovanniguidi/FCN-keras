import argparse
import os
import numpy as np
import json
import yaml

from data_generators.data_generator import DataGenerator
from models.model import ModelFCN
from trainers.trainer import TrainerFCN
from predictors.predictor import PredictorFCN
from utils.score_prediction import score_prediction
from preprocessing.preproc_functions import read_image, normalize_0_mean_1_variance


def train(args):
    """
    Train a model on the train set defined in labels.json
    """
    
    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f)
        
    with open(config['labels_file']) as f:
        dataset = json.load(f)
                
    train_generator = DataGenerator(config, dataset['train'], shuffle=True, 
                                    use_data_augmentation=config['data_aug']['use_data_aug'])
        
    #----------val generator--------
    val_generator = DataGenerator(config, dataset['val'], shuffle=True, use_data_augmentation=False)
    
    train_model = ModelFCN(config)
    trainer = TrainerFCN(config, train_model, train_generator, val_generator)

    trainer.train()

    
def predict_on_test(args):
    """
    Predict on the test set defined in labels.json
    """
        
    config_path = args.conf
    
    with open(config_path) as f:
        config = yaml.load(f)
        
    with open(config['labels_file']) as f:
        dataset = json.load(f)
        
    test_generator = DataGenerator(config, dataset['train'], shuffle=False, use_data_augmentation=False)
    
    #numpy array containing images
    images_test, labels_test = test_generator.get_full_dataset()

    #print(images_test.shape)
    #print(len(labels_test))
    
    graph_file =  config['network']['graph_path']
    weights_file = config['predict']['weights_file']
    batch_size = config['predict']['batch_size']
    
    predictor = PredictorFCN(config)
   
    pred_test = predictor.predict(images_test)

    pixel_accuracy, mean_accuracy, mean_IoU, freq_weighted_mean_IoU = score_prediction(labels_test, pred_test, 80)
    
#    for i in range(20):
#        print(labels_test[i], pred_test[i])
    
    print("pixel accuracy:", round(pixel_accuracy, 2))
    print("mean accuracy:", round(mean_accuracy, 2))
    print("mean IoU:", round(mean_IoU, 2))
    print("freq_weighted_mean_IoU:", round(freq_weighted_mean_IoU, 2))
    

def predict(args):
    """
    Predict on a single image 
    """
        
    config_path = args.conf

    filename = args.filename
    
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    with open(config['labels_file']) as f:
        dataset = json.load(f)
        
    test_generator = DataGenerator(config, dataset['test'], shuffle=False, use_data_augmentation=False)
        
    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    num_channels = config['image']['image_size']['num_channels']
    convert_to_grayscale = config['image']['convert_to_grayscale']
            
    #read image
    if num_channels == 1 or (num_channels == 3 and convert_to_grayscale):
        image = read_image_BW('./', filename, y_size, x_size)
        image = normalize_0_mean_1_variance_BW(image, y_size, x_size)
        image = np.reshape(image, (1, y_size, x_size, 1))
    else:
        image = read_image_color('./', filename, y_size, x_size)
        image = normalize_0_mean_1_variance_color(image, y_size, x_size)
        image = np.reshape(image, (1, y_size, x_size, 3))
    
    #print(image.shape)
        
    graph_file =  config['predict']['graph_file']
    weights_file = config['predict']['weights_file']
    batch_size = 1

    predictor = PredictorSeq2Seq(config, graph_file, weights_file, test_generator.num_decoder_tokens, test_generator.max_seq_length, test_generator.token_indices, test_generator.reverse_token_indices, 
                                 batch_size = batch_size)
   
    pred = predictor.predict(image)
    
    print("Predicted label:", pred[0])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Seq2seq')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)
    
#    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
#    subparsers.required = True
    
#    predict_parser = subparsers.add_parser('--predict')
#    predict_parser.add_argument('--filename', help='')

#    pascal_parser = subparsers.add_parser('pascal')
#    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    
#    argparser.add_argument('--predict-on-test', action='store_true', help='predict mode')
 #   parser = argparse.ArgumentParser(description='Arg parser')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')    
    group.add_argument('--predict_on_test', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')

    parser.add_argument('--filename', help='path to file')
    
    args = parser.parse_args()
   
    #    print(args)
      
    if args.predict_on_test:
        print('Predicting on test set')
        predict_on_test(args)      

    elif args.predict:
        if args.filename is None:
            raise Exception('missing --filename FILENAME')
        else:
            print('predict')
        predict(args)

    elif args.train:
        print('Starting training')
        train(args)       
    else:
        raise Exception('Unknown args') 
