import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from preprocessing import augment, preprocess
from model_evaluation import PerfEvalCustomCallback, evaluate_model


data = np.load("data/data_train.npy")
labels = np.load("data/labels_train.npy")

cnn_input, labels_new = augment(preprocess(data), labels)
cnn_input = cnn_input.reshape(-1,100,100,3)

x_train, x_inter, labels_train, labels_inter = train_test_split(cnn_input, labels_new, train_size=.7)
x_val, x_test, labels_val, labels_test = train_test_split(x_inter, labels_inter, train_size=.5)

x_train = x_train.reshape(-1, 100, 100, 3)
x_val = x_val.reshape(-1, 100, 100, 3)
x_test = x_test.reshape(-1, 100, 100, 3)

labels_train_ohe = OneHotEncoder().fit_transform(labels_train.reshape(-1,1)).toarray()
labels_val_ohe = OneHotEncoder().fit_transform(labels_val.reshape(-1,1)).toarray()
labels_test_ohe = OneHotEncoder().fit_transform(labels_test.reshape(-1,1)).toarray()

cnn_dataset = (x_train, labels_train_ohe, x_val, labels_val_ohe, x_test, labels_test_ohe)

# transfer learning using ImageNet weights used here (freeze all original hidden layer weights)
base_model = keras.applications.Xception(
weights='imagenet',
input_shape=(100,100,3),
include_top=False)
base_model.trainable=False

# add correctly sized input and output layers
inputs = keras.Input(shape=(100,100,3))
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(200, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=base_model.inputs, outputs=outputs)

# make trainable again after adding in necessary layers
for layer in model.layers:
    layer.trainable = True

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

perf_data = np.zeros((10, 4))
perf_eval_cb = PerfEvalCustomCallback(perf_data)
early_stop_cb = keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=4)

print(model.summary())

hobj = model.fit(x_train, labels_train_ohe, validation_data=(x_val, labels_val_ohe), epochs=10, batch_size=50,
         shuffle=True, callbacks=[perf_eval_cb, early_stop_cb], verbose=1)

eff_epochs = len(hobj.history['loss'])
eval_data = (perf_data[0:eff_epochs,:], cnn_dataset)

# shows loss and accuracy progression diagram
evaluate_model("test", model, eval_data)

model.save('models/final_trained_model')
