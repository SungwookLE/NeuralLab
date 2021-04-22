'''
##### Step 0. LOAD DATA
'''
import pickle

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

'''
##### Step 1: Dataset Summary & Exploration
'''
import numpy as np
import pandas as pd

n_label_train = pd.DataFrame(y_train)
n_label_train.drop_duplicates(inplace=True) 
train_info = {'num' : X_train.shape[0], 'size': X_train.shape[1:4] , 'n_class': len(n_label_train)}

n_label_valid = pd.DataFrame(y_valid)
n_label_valid.drop_duplicates(inplace=True)
valid_info = {'num': X_valid.shape[0], 'size': X_valid.shape[1:4], 'n_class': len(n_label_valid)}

n_label_test = pd.DataFrame(y_test)
n_label_test.drop_duplicates(inplace=True)
test_info = {'num': X_test.shape[0], 'size': X_test.shape[1:4], 'n_class': len(n_label_test)}

n_train = train_info['num']
n_validation = valid_info['num']
n_test = test_info['num']
image_shape = train_info['size']
n_classes = train_info['n_class']

print("Step 1: Dataset Summary & Exploration")
print("Number of training examples =", n_train)
print("Number of valid examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

index1 = random.randint(0, len(X_train))
image1 = X_train[index1].squeeze()

index2 = random.randint(0, len(X_train))
image2 = X_train[index2].squeeze()


fig, ax = plt.subplots(1,3)
fig.set_figwidth(15)

ax[0].set_title("label = {}".format(y_train[index1]))
ax[0].imshow(image1)

ax[1].set_title("label = {}".format(y_train[index2]))
ax[1].imshow(image2)

ax[2].grid()
ax[2].hist(y_train, color = 'skyblue', label = "Train Data")
ax[2].hist(y_test, color = 'green', label = "Test Data")
ax[2].hist(y_valid, color = 'orange', label = "Valid Data")
ax[2].set_title('Traffic Sign Data Count')
ax[2].legend()
ax[2].set_ylabel('Number of Instances')
ax[2].set_xlabel('Sample Count')

'''
##### Step 2: Design and Test a Model Architecture
'''
import cv2

def normalize_all(input):
    input_cpy = np.copy(input)
    input_cpy = input_cpy.astype(np.uint8)

    h,w,d = input_cpy[0].shape

    normal = np.zeros((len(input_cpy[:]), 32,32,1))
    for i in range (len(input_cpy[:])):
        gray_img = cv2.cvtColor(input_cpy[i], cv2.COLOR_RGB2GRAY)
        hist_equal = cv2.equalizeHist(gray_img).reshape(32,32,1)
        normal[i] = (hist_equal - 128.0) / (128.0)
    
    return normal

print(X_train.shape)
X_train_normal = normalize_all(X_train)
X_valid_normal = normalize_all(X_valid)
X_test_normal = normalize_all(X_test)

print(X_train_normal.shape)
plt.imshow(X_train_normal[0])

### Define architecture here.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def Conv_Net(input_feed, keep_prob=1):
    #input_feed shape = (32,32,1)
    mu = 0
    sigma = 0.1

    weights = {"conv_layer1": tf.Variable(tf.truncated_normal([2,2,1,6], mean=mu, stddev=sigma)), "conv_layer2": tf.Variable(tf.truncated_normal([5,5,6,16],                  mean=mu, stddev=sigma)) , "fc_layer3": tf.Variable(tf.truncated_normal([400,120])) , "fc_layer4": tf.Variable(tf.truncated_normal([120,90]))                   ,"fc_layer5": tf.Variable(tf.truncated_normal([90,43]))}
    biases  = {"conv_layer1": tf.Variable(tf.zeros(6)), "conv_layer2": tf.Variable(tf.zeros(16))
               ,"fc_layer3": tf.Variable(tf.zeros(120)), "fc_layer4":tf.Variable(tf.zeros(90))
               ,"fc_layer5": tf.Variable(tf.zeros(43))}

    # CONV_LAYER 1
    conv_layer1 = tf.nn.conv2d(input_feed, weights['conv_layer1'], strides=[1,1,1,1], padding = 'VALID')    
    conv_layer1 = tf.nn.bias_add(conv_layer1, biases['conv_layer1'])
    conv_layer1 = tf.nn.relu(conv_layer1)
    conv_layer1 = tf.nn.dropout(conv_layer1, keep_prob=keep_prob)

    # POOLING
    pool_layer1 = tf.nn.max_pool(conv_layer1, ksize=[1,2,2,1], strides=[1,2,2,1], padding ='VALID')

    # CONV_LAYER 2
    conv_layer2 = tf.nn.conv2d(pool_layer1, weights['conv_layer2'], strides =[1,1,1,1], padding = 'VALID' )
    conv_layer2 = tf.nn.bias_add(conv_layer2, biases['conv_layer2'])
    conv_layer2 = tf.nn.relu(conv_layer2)
    conv_layer2 = tf.nn.dropout(conv_layer2, keep_prob=keep_prob)

    # POOLING
    pool_layer2 = tf.nn.avg_pool(conv_layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # FLATTEN
    neural_feed = flatten(pool_layer2)

    # Fully_Connected Layer 3
    fc_layer3 = tf.matmul(neural_feed, weights['fc_layer3'])
    fc_layer3 = tf.nn.bias_add(fc_layer3, biases['fc_layer3'])
    fc_layer3 = tf.nn.relu(fc_layer3)
    fc_layer3 = tf.nn.dropout(fc_layer3, keep_prob=keep_prob)

    # Fully_Connected Layer 4
    fc_layer4 = tf.matmul(fc_layer3, weights['fc_layer4'])
    fc_layer4 = tf.nn.bias_add(fc_layer4, biases['fc_layer4'])
    fc_layer4 = tf.nn.relu(fc_layer4)
    fc_layer4 = tf.nn.dropout(fc_layer4, keep_prob=keep_prob)

    # Fully_Connected Layer 5
    fc_layer5 = tf.matmul(fc_layer4, weights['fc_layer5'])
    fc_layer5 = tf.nn.bias_add(fc_layer5, biases['fc_layer5'])
    fc_layer5 = tf.nn.relu(fc_layer5)
    fc_layer5 = tf.nn.dropout(fc_layer5, keep_prob=keep_prob)

    logits = fc_layer5 #needed softmax for clustering

    return logits, conv_layer1, conv_layer2

### Train 
x = tf.placeholder(tf.float32, (None, 32,32,1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32, (None))

rate =0.0015
logits, conv1, conv2 = Conv_Net(x,keep_prob)
Cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
Loss_mean = tf.reduce_mean(Cross_entropy)
Optimizer = tf.train.AdamOptimizer(learning_rate = rate)
Training_operation = Optimizer.minimize(Loss_mean)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict = {x: batch_x, y: batch_y, keep_prob :1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy/ num_examples

### RUN
EPOCHS = 10
BATCH_SIZE = 120

loss_set_forplot = []
accuracy_set_forplot=[]

saver=tf.train.Saver()
print("Step 2: Design and Test a Model Architecture")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Trainnig...")
    print()
    for i in range(EPOCHS):
        X_train_normal, y_train = shuffle(X_train_normal, y_train)
        loss_log = 0
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset+BATCH_SIZE
            batch_x, batch_y = X_train_normal[offset:end], y_train[offset:end]
            sess.run(Training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.9})
            loss_log += sess.run(Loss_mean, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.9})

        loss_set_forplot.append(loss_log)
        
        train_accuracy = evaluate(X_train_normal ,y_train)
        validation_accuracy = evaluate(X_valid_normal ,y_valid)
        accuracy_set_forplot.append([train_accuracy,validation_accuracy])
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    
    saver.save(sess, './mine')
    print("Model Saved")
    print()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test_normal, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

accuracy_set_forplot = np.array(accuracy_set_forplot)
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].plot(loss_set_forplot, 'o--', color = 'red', label='loss')

ax[1].plot(accuracy_set_forplot[:,0],'--o', color = 'red', label ='train accuracy')
ax[1].plot(accuracy_set_forplot[:,1],'--o', color= 'blue', label ='valid accuracy')

ax[0].grid()
ax[1].grid()
ax[0].set_xlabel('EPOCHS')
ax[1].set_xlabel('EPOCHS')

ax[0].legend()
ax[1].legend()

ax[0].set_title('Learning Loss')
ax[1].set_title('Learning Accuarcy')
plt.show()
fig.savefig('learn_monitor')

'''
##### Step 3: Test a Model on New Images
'''
import glob
import re

signames = pd.read_csv('signnames.csv')
new_images = glob.glob('./New_images/*.png')
regex = re.compile(r'\d{1,2}')

new_label=[]
for fname in new_images:
    match_obj = regex.search(fname)
    new_label.append(int(match_obj.group()))

i=0
new_set = np.zeros((len(new_label),32,32,3), dtype=np.uint8)
for fname in new_images:
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(32,32), interpolation=cv2.INTER_AREA)
    new_set[i] = np.array(img)
    i+=1

print("Step 3: Test a Model on New Images")

### Visualize your network's feature maps here.
# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    tf_activation=tf.convert_to_tensor(tf_activation)
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

#Using new_image
fig, ax = plt.subplots(1,2, figsize=(6,2))
i=0
for img in new_set:
    ax[i].imshow(img)
    ax[i].set_title("CLASS:" + str(new_label[i]) + ", " + signames["SignName"].iloc[new_label[i]])
    i+=1

plt.show()
X_shuffle = new_set
y_shuffle = new_label
table_res = pd.DataFrame( columns=['Predicted', 'Label', 'Logits(top_k)', 'Class(top_k)'] )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,'./mine')

    X_shuffle_normal = normalize_all(X_shuffle)
    logi = sess.run(logits, feed_dict={x: X_shuffle_normal, keep_prob: 1})
    visual_conv1 = sess.run(conv1, feed_dict={x:X_shuffle_normal, keep_prob:1})
    visual_conv2 = sess.run(conv2, feed_dict={x:X_shuffle_normal, keep_prob:1})

    validation_accuracy = evaluate(X_shuffle_normal, y_shuffle)
    TOP_logit = sess.run(tf.nn.top_k(logi, k=3))
    
    for i in range(len(X_shuffle_normal)):
        table_res.loc[i+1] = [ (signames["SignName"].iloc[np.argmax(logi[i],0)]),
                               (signames["SignName"].iloc[y_shuffle[i]]),
                               (TOP_logit[0][i].squeeze()),
                               (TOP_logit[1][i].squeeze())
                             ] 
        table_res.tail()
        outputFeatureMap(X_shuffle_normal[i][np.newaxis,:], visual_conv1[i][np.newaxis,:], activation_min=-1, activation_max=-1, plt_num = i*2 )
        outputFeatureMap(X_shuffle_normal[i][np.newaxis,:], visual_conv2[i][np.newaxis,:], activation_min=-1, activation_max=-1, plt_num = i*2+1 )
        
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(table_res)
    table_res.to_csv("results.csv")
    print()
    print("New Image Accuracy: {:.1f}".format(validation_accuracy*100), '% !')
