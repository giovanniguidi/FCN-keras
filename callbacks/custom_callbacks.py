import keras

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_train_end(self):
        print('train end')

    def on_batch_begin(self):
        print('batch begin')
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
    def on_epoch_begin(self):
        print('epoch begin')

    def on_epoch_end():
        print('epoch end')
        