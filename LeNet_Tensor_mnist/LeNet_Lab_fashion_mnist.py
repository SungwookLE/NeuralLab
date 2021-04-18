############################################################################################################ DATA_SET IMPORT: LOAD MNIST
import tensorflow as tf
import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]

print()
print("Image Shape: {}".format(X_train[0].shape))

print("Training Set: {} samples".format(len(X_train)))
print("Test Set: {} samples".format(len(X_test)))

############################################################################################################ Using Lenet, Data makes 32x32x1 from 28x28x1
# Pad images with 0(zeros)

X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

X_train = np.array(X_train, dtype = np.float32)
X_test = np.array(X_test, dtype = np.float32)

print("Updated Image Shape: {}".format(X_train[0].shape))

############################################################################################################ Visualize Random Data
import random
import matplotlib.pyplot as plt

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

#plt.imshow(image, cmap="gray")
#print(y_train[index])
#plt.show()

############################################################################################################ Process Data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

############################################################################################################ Setup TensorFlow
EPOCHS = 5
BATCH_SIZE = 120

############################################################################################################ LeNet Architecture Design
from tensorflow.contrib.layers import flatten
import tensor_monitor_custom

def LeNet(x):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1, Output = 28x28x6.
    weight_conv_layer1 = tf.Variable(tf.truncated_normal([2,2,1,6], mean = mu, stddev = sigma))
    bias_conv_layer1 = tf.Variable(tf.zeros(6))

    conv_layer1= tf.nn.conv2d(x, weight_conv_layer1, strides=[1,1,1,1], padding = 'VALID')
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

############################################################################################################ Features and Labels
'''
Train LeNet to classify MNIST data
x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels
'''
x = tf.placeholder(tf.float32, (None, 32,32,1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

############################################################################################################ Training Pipeline
'''
Create a training pipeline that uses the model to classify MNIST data
'''
rate = 0.001
logits = LeNet(x)
Cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(Cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

############################################################################################################ Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy/ num_examples

############################################################################################################ Train the MOodel
'''
Run the trainning data through the training pipeline to train the model
Before each epoch, shuffle thr trainning set.
After each epoch, measure the loss and accuracy of the validation set.
'''
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Trainnig...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset+BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_test ,y_test)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet_fashion')
    print("Model saved")


############################################################################################################ TESTING the Model
#테스트 이미지 한장만 샘플로 넣어서 이미지가 어떻게 학습되엇는지 결과 표출

image = X_train[index].squeeze()
plt.imshow(image, cmap="gray")

sample = X_train[index]
sample = sample[np.newaxis]
print(sample.shape)

def one_feed_sampling_test(X_data, index):
    sess = tf.get_default_session()
    raw_labels = sess.run(logits, feed_dict = {x: sample})
    predict_labels = sess.run(tf.argmax(raw_labels,1))

    return raw_labels, predict_labels

label_name ={ 0: "T-shirt", 1: "Pants", 2: "pull-over", 3: "dress", 4:"coat", 5: "sandle", 6:"shirt", 7:"snikerz", 8:"bag", 9:"boots"}

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet_fashion.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    raw_labels, predict_labels = one_feed_sampling_test(X_train, index)
    print("RESULT(LABELS): ", raw_labels.squeeze())
    print("PREDICT: ",label_name[predict_labels[0]])

############################################################################################################ Evaluate the Model
#TEST SET ALL EVALUATE

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('lenet_fashion.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

plt.show()

#https://tykimos.github.io/2018/09/30/Hello_Fashion_MNIST/
