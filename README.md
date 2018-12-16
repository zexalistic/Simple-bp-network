# Simple-bp-network
This is a homework of course ECE2191 

Neural Network for Handwritten Character Recognition

 

We will implement an Artificial Neural Network, which is trained using the back propagation algorithm, to recognize the handwritten characters.

 

Data can be downloaded from:

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz

Please use the image data.

 

1. In this project, the result of ANN is a matrix of 1*62. The output being obtained from the ANN can be used to obtain one of the 62 (26 Upper-Case, 26 Lower-Case, 10 numbers) characters of the English language.

2. The neural network implemented in this project is composed of 3 layers, one input, one hidden and one output. The input layer has neurons which use the images as inputs, the hidden layer has 100 neurons, and the output layer has 62 neurons.

3. For this project, the sigmoid function is used as a non-linear neuron activation function. Bias terms (equal to 1) with trainable weights were also included in the network structure.

4. Each class has 55 samples.  In each class, please randomly select 5 samples as testing data and the rest 50 samples as training data.
