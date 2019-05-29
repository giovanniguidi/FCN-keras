import keras

class BaseDataGenerator(keras.utils.Sequence):
    
    def __init__(self, config, shuffle, use_data_augmentation):
        """
        Base constructor
        """
        self.config = config
        self.dataset_folder = self.config['dataset_folder']
        self.batch_size = self.config['train']['batch_size']
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.shuffle = shuffle
        self.use_data_aug = use_data_augmentation
        
    def __len__(self):
        """Gives the number of batches per epoch
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """Returns a batch of data

        Parameters
        ------
        index: int
            index of the batch
            
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Function called when finishing an epoch

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def data_generation(self, dataset_temp):      
        """Read and normalize images of the batch 

        Parameters
        ------
        dataset_temp: list
            list of IDs of the elements in the batch
            
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError
