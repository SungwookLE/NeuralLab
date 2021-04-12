############################################################################################################ DATA_SET IMPORT: LOAD MNIST
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train            = mnist.train.images, mnist.train.labels
X_validation, y_validation  = mnist.validation.images, mnist.validation.labels
X_test, y_test              = mnist.test.images, mnist.test.labels

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set: {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set: {} samples".format(len(X_test)))
############################################################################################################ Using Lenet, Data makes 32x32x1 from 28x28x1
'''
The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
You do not need to modify this section.
'''
import numpy as np
# Pad images with 0(zeros)
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

############################################################################################################ Visualize Random Data
import random
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])
#plt.show()

############################################################################################################ Process Data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


############################################################################################################ Setup TensorFlow
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 120

############################################################################################################ Implement LeNet-5
# VERY GOOD REFERENCE: http://yann.lecun.com/exdb/lenet/

'''
Input:
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

**Architecture
Layer 1: Convolutional. The output shape should be 28x28x6.
Activation. Your choice of activation function.
Pooling. The output shape should be 14x14x6.
Layer 2: Convolutional. The output shape should be 10x10x16.
Activation. Your choice of activation function.
Pooling. The output shape should be 5x5x16.
Flatten. Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
Layer 3: Fully Connected. This should have 120 outputs.
Activation. Your choice of activation function.
Layer 4: Fully Connected. This should have 84 outputs.
Activation. Your choice of activation function.
Layer 5: Fully Connected (Logits). This should have 10 outputs.
Output
Return the result of the 2nd fully connected layer.
'''

from tensorflow.contrib.layers import flatten

def LeNet(x):
    mu = 0 
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1, Output = 28x28x6.
    weight_conv_layer1 = tf.Variable(tf.truncated_normal([5,5,1,6], mean=mu, stddev= sigma))
    bias_conv_layer1 = tf.Variable(tf.zeros(6))

    conv_layer1= tf.nn.conv2d(x,weight_conv_layer1, strides=[1,1,1,1], padding = 'VALID')
    conv_layer1= tf.nn.bias_add(conv_layer1, bias_conv_layer1)
    conv_layer1 = tf.nn.relu(conv_layer1) #Activation: Relu

    # Sub: Pooling. Input = 28x28x6, Output = 14x14x6
    pool_layer1 = tf.nn.max_pool(conv_layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding ='VALID')

    # Layer 2: Convolutional, Input = 14x14x6, Output = 10x10x16.
    weight_conv_layer2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev= sigma))
    bias_conv_layer2 = tf.Variable(tf.zeros(16))

    conv_layer2= tf.nn.conv2d(pool_layer1, weight_conv_layer2, strides=[1,1,1,1], padding='VALID')
    conv_layer2= tf.nn.bias_add(conv_layer2, bias_conv_layer2)
    conv_layer2= tf.nn.relu(conv_layer2) #Activation: Relu

    # Sub: Pooling. Input = 10x10x16, Output = 5x5x16
    pool_layer2 = tf.nn.avg_pool(conv_layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Flatten To Fully Connected Layer. Input = 5x5x16, Output = 400
    neural_feed = flatten(pool_layer2)

    # Layer 3: Fully Connected, Input = 400, Output = 120
    weight_fc_layer3 = tf.Variable(tf.truncated_normal([400,120], mean=mu, stddev=sigma))
    bias_fc_layer3 = tf.Variable(tf.zeros(120))

    fc_layer3 = tf.matmul(neural_feed, weight_fc_layer3)
    fc_layer3 = tf.nn.bias_add(fc_layer3, bias_fc_layer3)
    fc_layer3 = tf.nn.relu(fc_layer3)

    # Layer 4: Fully Connected, Input = 120, Output = 84
    weight_fc_layer4 = tf.Variable(tf.truncated_normal([120,84], mean=mu, stddev=sigma))
    bias_fc_layer4 = tf.Variable(tf.zeros(84))

    fc_layer4 = tf.matmul(fc_layer3, weight_fc_layer4)
    fc_layer4 = tf.nn.bias_add(fc_layer4, bias_fc_layer4)
    fc_layer4 = tf.nn.relu(fc_layer4)

    # Layer 5: Fully Connected, Input = 84, Output = 10
    weight_fc_layer5 = tf.Variable(tf.truncated_normal([84,10], mean=mu, stddev=sigma))
    bias_fc_layer5 = tf.Variable(tf.zeros(10))

    fc_layer5 = tf.matmul(fc_layer4, weight_fc_layer5)
    fc_layer5 = tf.nn.bias_add(fc_layer5, bias_fc_layer5)

    logits = fc_layer5

    return logits

print(LeNet(X_train).shape)
