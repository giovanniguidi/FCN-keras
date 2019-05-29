class BaseTrain(object):
    """
    Base class for training a model

    Attributes
    ----------
    config : dict
        configuration file 
    model : keras.models 
        dataset list, each elemen is a dictionary {'filename':, 'label'}
    train_generator : DataGenerator
        generator of the train set
    val_generator : DataGenerator
        generator of the val set

    Methods
    -------
    train()
        train a model
    save_model(model, graph_path, weights_path)
        save a model
    """
    
    def __init__(self, config, model, train_generator, val_generator):
        """
        Base constructor
        """
        self.config = config
        self.model = model.model
        self.train_generator = train_generator
        self.val_generator = val_generator
        
    def train(self):
        """Train a model

        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError

    def save_model(self, model, graph_path, weights_path):
        """Save a model

        Parameters
        ------
        model: keras.models
            keras model
        graph_path: str
            path to save the graph
        weights_path: str
            path to save the weights
            
        Raises
        ------
        NotImplementedError
        """
        
        raise NotImplementedError
