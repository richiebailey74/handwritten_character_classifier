import tensorflow as tf
import keras
from .train import plot_training_perf

def evaluate_model(name, model, eval_data,
                   plot_training=True, evaluate_on_test_set=True):
    # unpack the stuff
    perf_data, dataset = eval_data
    train_x, train_y, val_x, val_y, test_x, test_y = dataset

    # get predictions from the model
    train_preds = model.predict(train_x)
    val_preds = model.predict(val_x)

    # measure the accuracy (as categorical accuracy since we have a softmax layer)
    catacc_metric = keras.metrics.CategoricalAccuracy()
    catacc_metric.update_state(train_y, train_preds)
    train_acc = catacc_metric.result()

    catacc_metric = keras.metrics.CategoricalAccuracy()
    catacc_metric.update_state(val_y, val_preds)
    val_acc = catacc_metric.result()
    print('[{}] Training Accuracy: {:.3f}%, Validation Accuracy: {:.3f}%'.format(name, 100 * train_acc, 100 * val_acc))

    if plot_training:
        plot_training_perf(perf_data[:, 0], perf_data[:, 1], perf_data[:, 2], perf_data[:, 3])

    if evaluate_on_test_set:

        test_preds = model.predict(test_x)

        accuracy_obj = keras.metrics.CategoricalAccuracy()
        accuracy_obj.update_state(test_y, test_preds)
        test_acc = accuracy_obj.result()

        test_loss, _ = model.evaluate(test_x, test_y)  # the test_acc could also be extracted here where the _ is

        cat_loss_obj = tf.keras.losses.CategoricalCrossentropy()
        test_ce_loss = cat_loss_obj(test_y, test_preds)

        print('[{}]  Test loss: {:.5f}; test accuracy: {:.3f}%'.format(name, test_loss, 100 * test_acc))
        print('[{}]  Test cross entropy loss: {:.5f}'.format(name, test_ce_loss))

    return