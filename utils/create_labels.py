import json
from sklearn.model_selection import train_test_split
import os
from scipy.io import loadmat

def create_labels(data_folder, test_size, val_size, random_state):

    #training set
    images_folder = './JPEGImages/'
    annotations_folder = './SegmentationClass/'
#    annotations_folder = './SegmentationObject/'
 
    annotations = os.listdir(data_folder + annotations_folder)

#    print(len(annotations))
    
    dataset = []
    
    for annotation in annotations:
        out_dict = {}
        out_dict['annotation'] = annotations_folder + annotation
        out_dict['filename'] = images_folder + annotation[:-3] + 'jpg'
#        out_dict['annotation'] = annotations_train_folder + image[:-3] + 'png'
        dataset.append(out_dict)
       
#    print(len(dataset))
        
    dataset_train_val, dataset_test = train_test_split(dataset, test_size=test_size, random_state=random_state)  
    dataset_train, dataset_val = train_test_split(dataset_train_val, test_size=val_size, random_state=random_state)  
        
    
#    class_names = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Dining-table', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tv-monitor']

    class_names = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Dining-table', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tv-monitor']

#    print(len(class_names))        
        
    labels_dict = {key: value for (key, value) in enumerate(class_names)}        
        
#    print(labels_dict)

    #-------dataset out
    dataset_out = {}

    dataset_out['train'] = dataset_train
    dataset_out['val'] = dataset_val
    dataset_out['test'] = dataset_test
    dataset_out['labels'] = labels_dict

    return dataset_out



if __name__ == '__main__':

    
    data_folder = '../datasets/VOCtrainval_25-May-2011/TrainVal/VOCdevkit/VOC2011/'
    filename_out = '../datasets/labels.json'
    
    #split validation into val and test
    test_size = 0.1
    val_size = 0.1
    
    random_state = 18
    
    dataset_out = create_labels(data_folder, test_size = test_size, val_size = val_size, random_state = random_state)
    
#    print(len(dataset_out['train']))

#    print(dataset_out['train'][0])

    
#    print(dataset_out['labels'])    
#    print(len(dataset_out['test']))
    
    with open(filename_out, 'w') as f:
        json.dump(dataset_out, f)
    