# Classifying MNIST and CIFAR10 with Keras using Googlenet, ResNet34 and VGG16

The goal of this project is to building some milestone Deep Convolutional Neural Networks from scratch and using them for classification. 

## Dataset

<img src="/imgs/mnist.png" width="400" alt="MNIST">
<img src="/imgs/cifar10.png" width="400" alt="CIFAR10">

There are two dataset being used for that experiment. [MNIST](https://keras.io/api/datasets/mnist/#load_data-function) and [CIFAR10](https://keras.io/api/datasets/cifar10/#load_data-function)

 1) MNIST is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images. More info can be found at the [MNIST homepage] (http://yann.lecun.com/exdb/mnist/).
 2) CIFAR10 is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. 


## Approach
I've carried out two tasks: 

1) Creating Googlenet and Resnet34 architectures from scratch and training on MNIST dataset.

2) Performed transfer learning. VGG16 model have been initialized with the weights based on a training on ImageNet dataset without including the top layer. Then new fully-connected layer with random initialization, with the correct number of output. 
Just new fully-connected layer has been trained.


## Results

| Model  | Accuracy |
| ------------- | ------------- |
| 1. GoogleNet model  | 98.07% |
| 2. ResNet34 model  | 98.23% |
| 3. VGG16 model  | 65.10%  | 

VGG16 model accuracy can also be pulled up by increasing the number of epoch. 
