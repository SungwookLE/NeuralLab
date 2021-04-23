import pickle
import numpy as np
import tensorflow as tf

training_file = "./train.p"
validation_file= "./valid.p"
testing_file = "./test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# TODO: Build the Final Test Neural Network in Keras Here
model=Sequential()
model.add( Conv2D(input_shape=(32,32,3), kernel_size=(3,3), filters=32, strides=(1,1), padding='VALID') )
model.add(Dropout(rate=0.5))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='VALID'))
model.add(Activation('relu'))
print("Conv1 out shape:",model.output_shape)

model.add( Conv2D(input_shape=(29,29,32), kernel_size=(5,5), filters=15, strides=(1,1), padding='SAME') )
model.add(Dropout(rate=0.5))
model.add(MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='SAME'))
model.add(Activation('relu'))
print("Conv2 out shape:",model.output_shape)

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(43))
model.add(Activation('softmax'))
print("out shape:",model.output_shape)

print(model.summary())
# preprocess data
from sklearn.preprocessing import LabelBinarizer
X_normalized = np.array(X_train / 255.0 - 0.5 )
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y_train)

# compile and fit the model
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
history = model.fit(X_normalized, y_one_hot, epochs=5, batch_size=120, validation_split=0.2)

# evaluate model against the test data
X_normalized_test = np.array(X_test / 255.0 - 0.5 )
y_one_hot_test = label_binarizer.fit_transform(y_test)

print("Testing")
metrics = model.evaluate(X_normalized_test, y_one_hot_test)
for metric_i in range(len(model.metrics_names)):
    metric_name = model.metrics_names[metric_i]
    metric_value = metrics[metric_i]
    print('{}: {}'.format(metric_name, metric_value))   