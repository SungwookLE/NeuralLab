# STUDY: Convolutional Neural Network
> **AUTHOR**: SungwookLE (joker1251@naver.com)  
> DATE: '21.4/27

### Reference
[1] [Convolution Processing](http://www.songho.ca/dsp/convolution/convolution.html)
<img src="http://www.songho.ca/dsp/convolution/files/conv2d_result.png" width="900" height="200">  
[2] LENET ARCHITECTURE [LENET DOCUMENT](http://yann.lecun.com/exdb/lenet/)  
<img src="https://video.udacity-data.com/topher/2016/November/581a64b7_arch/arch.png" width="70%">
<img src="http://yann.lecun.com/exdb/lenet/gifs/asamples.gif" width="900" height="200">
[3] Andrej Karpathy's [Standford CNN OCW](https://cs231n.github.io/)
[4] Michael Neilsen's [free book](http://neuralnetworksanddeeplearning.com/)
[5] Goodfellow, Bengio, and Courville's more advanced [free book](https://www.deeplearningbook.org/) on Deep Learning
[6] Paper [Visualizing and Understanding Deep Neural Networks](https://arxiv.org/abs/1311.2901)
[7] Official Tensorflow [homepage](https://www.tensorflow.org/tutorials)

### HOW TO EXECUTE
* Dependencies: This lab requires(Conda environment)`-`[CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
- (1) Number Classifier: `python LeNet_Lab_mnist.py`
- (2) Cloth Classifier: `python LeNet_Lab_fashion_mnist.py`

## [1] Covolution Basic
Convolution is fundamental concept in signal processing for various fields..
* Signal Filter: 1D Convolution
$$1D: y[n]=x[n]*h[n]= \Sigma^{inf}_{k=-inf}x[k]\cdot h[n-k]$$
1D conv is useful for analyzing the system characteristic such as differential equation. Because h[n] represents the system behavior, thus can be designed mathmatical filter using convolution.  
See the below example.  
![calc1](./conv1d_calc.png)
```c++
//pseudo code
for ( i = 0; i < sampleCount; i++ ) //sample means x[n]
{
    y[i] = 0;  // set to zero before sum
    for ( j = 0; j < kernelCount; j++ ) //kernel means h[n]
    {
        y[i] += x[i - j] * h[j]; // convolve: multiply and accumulate
    }
}
```
* Image Filter: Two dimensions convolution
$$2D: y[m,n] = x[m,n]*h[m,n]=\Sigma_{j=-inf}^{inf}\Sigma_{i=-inf}^{inf}x[i,j]\cdot h[m-i, n-j]$$
2D conv is useful for image filtering such as smoothie filter(Gaussian). Adjusting the h[m,n] coefficients, the image x[m,n] can be affected to each pixel through the convolution. i.e. h[m,n] is image filter system.  
![Convolution Matrix Operation](https://mblogthumb-phinf.pstatic.net/20160909_194/cjh226_1473416748613KENU3_GIF/Convolution_schematic.gif?type=w2)

See the below example.
![calc2](./conv2d_calc.png)

```c++
//pseudo code
// find center position of kernel (half of kernel size)
kCenterX = kCols / 2;
kCenterY = kRows / 2;

for(i=0; i < rows; ++i)              // rows
{
    for(j=0; j < cols; ++j)          // columns
    {
        for(m=0; m < kRows; ++m)     // kernel rows
        {
            mm = kRows - 1 - m;      // row index of flipped kernel

            for(n=0; n < kCols; ++n) // kernel columns
            {
                nn = kCols - 1 - n;  // column index of flipped kernel

                // index of input signal, used for checking boundary
                ii = i + (kCenterY - mm);
                jj = j + (kCenterX - nn);

                // ignore input samples which are out of bound
                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
                    out[i][j] += in[ii][jj] * kernel[mm][nn];
            }
        }
    }
}
```
* Example of convolution image filter(=kernel)
![conv filter kinds](https://mblogthumb-phinf.pstatic.net/20160909_263/cjh226_14734166023033zQk0_PNG/fig2.png?type=w2)

## [2] Convolutional Neural Network(CNN)
- [CNN Processing](https://www.youtube.com/watch?v=ISHGyvsT0QY)
![network overview](https://mblogthumb-phinf.pstatic.net/20160909_182/cjh226_1473414664622BSdJS_PNG/fig1.png?type=w2)
Image features are extracted through the convolution layer. With various techniques to make image feature more robust(anti over-fit) and better recognized(feature well-extracted), overall CNN network is working.

- CNN has the convolution layer that has two special things. One is `Feature Extraction`. Another one is `Statistical Invariance`.

1. Feature Extraction
    [![image](https://video.udacity-data.com/topher/2016/November/5837811f_screen-shot-2016-11-24-at-12.09.24-pm/screen-shot-2016-11-24-at-12.09.24-pm.png)](https://www.youtube.com/watch?v=ghEmQSxT6tw)
 A visualization of the third layer in the CNN. The gray grid on the left represents how this layer of the CNN activates (or "what it sees") based on the corresponding images from the grid on the right.
 Convolution Layer Weight Parameter works as image filter and this filter coefficients(parameters) are learned while CNN back-propagation process. In other words, this CNN architecture can extract image feature while learning convolution layer parameter.

2. Statistical Invariance(Weight Parameter Sharing)
    The weights, w, are shared across patches for a given layer in a CNN to detect the cat above regardless of where in the image it is located.
    <img src="https://video.udacity-data.com/topher/2016/November/58377f77_vlcsnap-2016-11-24-16h01m35s262/vlcsnap-2016-11-24-16h01m35s262.png" width=50%>
    If we want a cat that’s in the top left patch to be classified in the same way as a cat in the bottom right patch, we need the weights and biases corresponding to those patches to be the same, so that they are classified the same way.
    This is exactly what we do in CNNs. The weights and biases we learn for a given output layer are shared across all patches in a given input layer. Note that as we increase the depth of our filter, the number of weights and biases we have to learn still increases, as the weights aren't shared across the output channels.
    Sharing parameters not only helps us with translation invariance, but also gives us a smaller, more scalable model.

- Convolution Layer Operation
<img src="https://video.udacity-data.com/topher/2016/November/58377d67_vlcsnap-2016-11-24-15h52m47s438/vlcsnap-2016-11-24-15h52m47s438.png" width="50%">

|**Dimensionality Study**||
|:---|:---|
|**1) Given**||
|Input layer size| Width=W, Height=H|
|Convolution filter size| F|
|Stride| S|
|Padding| P|
|Number of filters| K|
|**2) Return**||
|Output layer size| W_out=[(W-F+2P)/S]+1, H_out=[(H-F+2P)/S]+1|
|Output Depth| D_out=K|
|Output Volume| W_out * H_out * D_out|

```python
# Tensorflow Framework example
# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME') 
# strides = [input_batch(ea), input_height, input_width, input_channels]
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```


|Example: Efficiency of CNN parameter sharing|
|---|
|**1. Setup**|
| H = height, W = width, D = depth |
| We have an input of shape 32x32x3 (HxWxD) |
| 20 filters of shape 8x8x3 (HxWxD) |
| A stride of 2 for both the height and width (S) |
| Zero padding of size 1 |
| Output Layer: 14x14x20 (HxWxD) |
| **2. RESULTS** |
|**2-1. Parameter Sharing (CNN method: Parameter Sharing)**|
|Parameter Number: (8x8x3+1)x20|
|**2-2. Without Parameter Sharing**|
|Parameter Number: (8x8x3+1)x(14x14x20)|

- Technique (Pooling, Dropout,  1x1 Convolution, Inception , ... )
1. Dropout: (for anti over-fit)
    [Lecture Youtube pt1](https://www.youtube.com/watch?v=6DcImJS8uV8&t)
    [Lecture Youtube pt2](https://www.youtube.com/watch?v=8nG8zzJMbZw)

    ```python 
    hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
    hidden_layer = tf.nn.relu(hidden_layer)
    hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
    # keep prob usually set as 0.5~0.8 for trainning process
    ```
    1-1. L2 Regularization (for parameter regularization)
[Lecture Youtube](https://img.youtu.be/QcJBhbuCl5g?t=56)
    
    1-2. Learning rate scheduling

2. Pooling: (for preventing over-fit & decreasing the size of the output)
    ![pooling](https://video.udacity-data.com/topher/2016/November/582aac09_max-pooling/max-pooling.png)
    Conceptually, the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.
  ```python
  ...
  conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
  conv_layer = tf.nn.bias_add(conv_layer, bias)
  conv_layer = tf.nn.relu(conv_layer)
  # Apply Max Pooling
  conv_layer = tf.nn.max_pool(
      conv_layer,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME')
  # ksize, strides = [batch, height, width, channels]
  # also, there is tf.nn.avg_pool()
  ```

3. 1x1 Convolution: For dimension reduction
    Nothing change to width and height, but adjusting(reduce) the depth of conv-filter, can reduce the conv parameter
    ![bottleneck](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99EE3E3359DC851929)

4. Inception Module(from GoogLeNet)
    Not choosing the one specific operation step, Choosing the all: [Inception](https://www.youtube.com/watch?v=SlTm03bEOxA&t)

  ![inception](https://user-images.githubusercontent.com/25279765/35002290-8af62246-fb2c-11e7-8692-12b9ccfff216.jpg)![image2](https://user-images.githubusercontent.com/25279765/35002517-441c8166-fb2d-11e7-9b40-b4216256cbb0.jpg)
  (b) is can reduce the dimension using 1x1 convolution: [REF.](https://kangbk0120.github.io/articles/2018-01/inception-googlenet-review)


## [3] Using Tensorflow, Implemente the Architecture, Traing Pipeline with Tensorflow Session(Graph)
First of all, there are many helpful documents such as [official tutorial](https://www.tensorflow.org/tutorials), [tf1 github repo](https://github.com/tensorflow/docs/tree/master/site/en/r1).

* tf.Session: [Check Here](https://chan-lab.tistory.com/6)
Session is gate to enter the tensorflow framework. When running the tensor framework, define the `tf.Session()` or `tf.InteractiveSession()` up to purpose.

* tf.Variable , tf.Constant, tf.placeholder: 
Weight, bias parameters must be defined as `tf.Variable`. `tf.Variable` makes those variables are learned while training process.  
`tf.Constatns` is method to define constant valus in tensor
`tf.placeholder` defines the feed data to architecture.
```python
#sample
x = tf.placeholder(tf.float32, (None, 32,32,1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10) #this is for one-hot encoding
```
* TF training pipeline sample
```python
#sample
rate = 0.001
logits = LeNet(x)
Cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(Cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

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
```  
  
* Deep Learning Architecture Sample
```python
#sample
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
```

- (1) Number Classifier: `python LeNet_Lab_mnist.py` or `execute LeNet_studying.ipynb`
(`ipynb` file will be easier to execute step by step...)
- (2) Cloth Classifier: `python LeNet_Lab_fashion_mnist.py`

## Next Study Subject(More Practice):
 1) Keras with traffic-sign classifier
 2) Behavior-clonning (Steering angle regression)