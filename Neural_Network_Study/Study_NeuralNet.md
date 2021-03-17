# Study of Neural-Net, Network

> **AUTHOR**: SungwookLE (joker1251@naver.com)  
> **DATE**: '21.3/17  

### Reference
* Project Explore [Github](https://github.com/SungwookLE/ReND_Car_TensorLab_with_NeuralNet)
* Machine Learning Notebook(#Theory) [Blog](https://sites.google.com/site/machinelearningnotebook2/home) 
* Deep Learning Term Database(#Theory) [Blog](http://www.aistudy.co.kr/neural/nn_definition.htm) 
* Neural Network BackPropagation1(#Practice) [Blog](https://excelsior-cjh.tistory.com/171)
* Neural Network BackPropagation2(#Practice) [Blog](https://bskyvision.com/718)


## [1] Linear Regression/Classification Problem
In the beginning, there were linear analysis methods. And those are *still useful and simple to use*, but, the linear models have limitation to increase their performance. 

* Linear Regression, [Linear_Regression](https://datalabbit.tistory.com/49)
<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbqu78G%2FbtqIVIXJgey%2Fn2ikobDvKDwdPbsSmQHrnk%2Fimg.png" width="80%" height="80%"></center>  

* Linear Classifier: Rogistic Regression, [Linear_Classification](https://mylearningsinaiml.wordpress.com/linear-classification/)
<center><img src="https://mylearningsinaiml.files.wordpress.com/2018/05/linearclassifier2.jpg?w=625" width="80%" height="80%"></center>  


## [2] Non-linear Regression/Classification Problem
Those method can *express the non-linearity about the data, and choose the region* though it would be divided each others with complexity.

* Nonlinear Regression, [Polynomial_Regression](https://wikidocs.net/34065)
<center><img src="https://wikidocs.net/images/page/34065/TRAIN_POLY.jpeg" width="80%" height="80%"></center>

* Nonlinear Classification: It can be described with [Perceptron]()
<center><img src="https://lh3.googleusercontent.com/proxy/Z8Z4CIWtTEPEmcJMRYlvrozoKmka7q_Er4XQZAuehbWKUHz0uTa-cJY3iPYGAc7CW6QdSrma3GwtzJWZZVfoQeJc2Dkm-gTOh3xQjjorOQHnDlBHBSYaTTtnqMY6eEmgcsEqzzJxj67bNsOMA0m3cr1NoGkpiNr0eOa7kKZ4y1vAYyI" width="80%" height="80%"></center>

### [2-1] Perceptron
<center><img src="https://image.slidesharecdn.com/lecture29-convolutionalneuralnetworks-visionspring2015-150504114140-conversion-gate02/95/lecture-29-convolutional-neural-networks-computer-vision-spring2015-9-638.jpg?cb=1430740006" width="80%" height="80%"></center>

* Equation of Single perceptron (Single Neuron)
$Output=f(A*inputs+B) \\
<where,\; f(x)\; is\; nonlinear\; function\; e.g.\; sigmoid,\; relu\; ...>$

Multi Perceptron Layer can describe the nonlinear classifier.
<center><img src="https://missinglink.ai/wp-content/uploads/2018/11/multilayer-perceptron.png" width="80%" height="80%"></center>

* Forward Propagation 

* Backward Propagation

## [3] Practice: Equation derivation and Code



