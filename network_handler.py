import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from data import get_data
from DenseLayer import DenseLayer
from ActivationReLU import ActivationReLU
from ActivationSoftMax import ActivationSoftMax
from Cost import CrossEntropy



class NeuralNetwork:
    def __init__(self):
        self.layers = []  

    def add_layer(self, layer):
        self.layers.append(layer)

    def save(self, filename):
        """Save the model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename):
        """Load the model from a file."""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    
    def train(self, images, labels, learning_rate=0.001,accuracy_check_number=5000, epochs=7):


        self.add_layer(DenseLayer(784,64)) #inputlayer        0
        self.add_layer(DenseLayer(64,32)) #hiddenlayer1       1
        self.add_layer(DenseLayer(32,10)) #hiddenlayer2       2
        self.add_layer(ActivationReLU()) #inputReLU           3
        self.add_layer(ActivationReLU()) #hiddenlayer1ReLU    4
        self.add_layer(ActivationSoftMax()) #outputSoftMax    5
        self.add_layer(CrossEntropy()) #CrossE                6


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
                self.layers[0].forward(images[j])
                self.layers[3].forward(self.layers[0].output)

                self.layers[1].forward(self.layers[3].output)
                self.layers[4].forward(self.layers[1].output)

                self.layers[2].forward(self.layers[4].output)
                self.layers[5].forward(self.layers[2].output)

                self.layers[6].forward(self.layers[5].output, labels[j])
                epoch_loss += self.layers[6].output
                # Backward pass
                self.layers[6].backward(self.layers[5].output, labels[j])

                self.layers[2].backward(self.layers[6].delta_output,learning_rate)
                self.layers[4].backward(self.layers[2].delta_output)
                self.layers[1].backward(self.layers[4].delta_output, learning_rate)

                self.layers[3].backward(self.layers[1].delta_output)
                self.layers[0].backward(self.layers[3].delta_output, learning_rate)

            
            #Accuracy check of current model
            for k in range(accuracy_check_number):
                self.layers[0].forward(images[k])
                self.layers[3].forward(self.layers[0].output)

                self.layers[1].forward(self.layers[3].output)
                self.layers[4].forward(self.layers[1].output)

                self.layers[2].forward(self.layers[4].output)
                self.layers[5].forward(self.layers[2].output)
                
                #convert softmax to one-hot
                ypred_onehot = np.zeros_like(self.layers[5].output, dtype=int)
                ypred_onehot[np.argmax(self.layers[5].output)] = 1
                
                if (ypred_onehot == labels[k]).all():
                    accuracy += 1

            
            average_accuracy = accuracy / accuracy_check_number
            average_loss = epoch_loss / (len(images) - accuracy_check_number)

            print(f"Epoch {i + 1}/{epochs}")
            print(f"Accuracy: {average_accuracy:.4f}")
            print(f"Loss: {average_loss:.4f}")
            
        self.save('model.pkl')
            
    def run(self, image_path):
        #network model load
        nn = self.load('model.pkl')
        #load new image to test against model!
        image = cv2.imread(f"testing_images/{image_path}")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (28, 28))
        #flip colours
        flat = resized_image.flatten() / 255
        flat_flip = 1-flat
        
        #forward prop
        nn.layers[0].forward(flat_flip)
        nn.layers[3].forward(nn.layers[0].output)

        nn.layers[1].forward(nn.layers[3].output)
        nn.layers[4].forward(nn.layers[1].output)

        nn.layers[2].forward(nn.layers[4].output)
        nn.layers[5].forward(nn.layers[2].output)
        
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title(f"The model predicts: {np.argmax(nn.layers[5].output)}")
        print(nn.layers[5].output)
        plt.show()
        
        
        
        
        
        
        
        
    
            