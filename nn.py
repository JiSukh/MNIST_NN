import numpy as np
from data import get_data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy

images, labels = get_data()
learning_rate = 0.001
accuracy_check_number = 5000

epochs = 10

#create new neural network

inputlayer = DenseLayer(784,12)
hiddenlayer1 = DenseLayer(12,12)
hiddenlayer2 = DenseLayer(12,10)
inputReLU = ActivationReLU()
hiddenlayer1ReLU = ActivationReLU()
outputSoftMax = ActivationSoftMax()
CrossE = CrossEntropy()



for i in range(epochs):
    epoch_loss = 0
    accuracy = 0
    

    #Shuffle training set, keep 5000 samples to use against network
    zipped = list(zip(images, labels))
    np.random.shuffle(zipped)
    images, labels = zip(*zipped)
    
    #Training loop
    for j in range(len(images)-accuracy_check_number):
        #forward pass
        inputlayer.forward(images[j])
        inputReLU.forward(inputlayer.output)

        hiddenlayer1.forward(inputReLU.output)
        hiddenlayer1ReLU.forward(hiddenlayer1.output)

        hiddenlayer2.forward(hiddenlayer1ReLU.output)
        outputSoftMax.forward(hiddenlayer2.output)

        CrossE.forward(outputSoftMax.output, labels[j])
        epoch_loss += CrossE.output
        # Backward pass
        CrossE.backward(outputSoftMax.output, labels[j])

        hiddenlayer2.backward(CrossE.delta_output,learning_rate)
        hiddenlayer1ReLU.backward(hiddenlayer2.delta_output)
        hiddenlayer1.backward(hiddenlayer1ReLU.delta_output, learning_rate)

        inputReLU.backward(hiddenlayer1.delta_output)
        inputlayer.backward(inputReLU.delta_output, learning_rate)

    
    #Accuracy check of current model
    for k in range(accuracy_check_number):
        inputlayer.forward(images[k])
        inputReLU.forward(inputlayer.output)

        hiddenlayer1.forward(inputReLU.output)
        hiddenlayer1ReLU.forward(hiddenlayer1.output)

        hiddenlayer2.forward(hiddenlayer1ReLU.output)
        outputSoftMax.forward(hiddenlayer2.output)
        
        #convert softmax to one-hot
        ypred_onehot = np.zeros_like(outputSoftMax.output, dtype=int)
        ypred_onehot[np.argmax(outputSoftMax.output)] = 1
        
        if (ypred_onehot == labels[k]).all():
            accuracy += 1



    print(hiddenlayer1.weights)
    print(CrossE.output)
    
    average_accuracy = accuracy / accuracy_check_number
    average_loss = epoch_loss / (len(images) - accuracy_check_number)

    print(f"Epoch {i + 1}/{epochs}")
    print(f"Accuracy: {average_accuracy:.4f}")
    print(f"Loss: {average_loss:.4f}")
