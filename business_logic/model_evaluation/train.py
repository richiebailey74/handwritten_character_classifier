import numpy as np
import matplotlib.pyplot as plt
import keras

class PerfEvalCustomCallback(keras.callbacks.Callback):

    def __init__(self, perf_data):
        self.perf_data = perf_data

    # we define the on_epoch_end callback and save the loss and accuracy in perf_data
    def on_epoch_end(self, epoch, logs=None):
        self.perf_data[epoch ,0] = logs['loss']
        self.perf_data[epoch ,1] = logs['accuracy']
        self.perf_data[epoch ,2] = logs['val_loss']
        self.perf_data[epoch ,3] = logs['val_accuracy']

    def get_perf_data(self):
        return self.perf_data


def plot_training_perf(train_loss, train_acc, val_loss, val_acc, fs=(8, 5)):
    plt.figure(figsize=fs)

    assert train_loss.shape == val_loss.shape and train_loss.shape == val_acc.shape and val_acc.shape == train_acc.shape

    # assume we have one measurement per epoch
    num_epochs = train_loss.shape[0]
    epochs = np.arange(0, num_epochs)

    plt.plot(epochs - 0.5, train_loss, 'm', linewidth=2, label='Loss (Training)')
    plt.plot(epochs - 0.5, train_acc, 'r--', linewidth=2, label='Accuracy (Training)')

    plt.plot(epochs, val_loss, 'g', linewidth=2, label='Loss (Validation)')
    plt.plot(epochs, val_acc, 'b:', linewidth=2, label='Accuracy (Validation)')

    plt.xlim([0, num_epochs])
    plt.ylim([0, 1.05])

    plt.legend()
    plt.show()
