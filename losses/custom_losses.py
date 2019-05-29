#import tensorflow as tf
#import keras
import keras.backend as K

def custom_categorical_crossentropy():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):

        #axis = 3 is softmax
        loss_val = K.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=3)
        
        #axis [1, 2] is img_dims
        mean_loss = K.mean(loss_val, axis=[1, 2])

#        mean_loss = tf.math.reduce_mean(loss, axis=None, keepdims=None, name=None)
#        mean_loss = Kmean(x, axis=None, keepdims=False):

        return mean_loss

    # Return a function
    return loss
        
        
        