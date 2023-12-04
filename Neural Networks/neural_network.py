import os
import numpy as np
import pandas as pd
from random import random

class Neural_Network:
    def create_network(self, number_of_inputs=4, number_of_hidden_neurons=4, number_of_outputs=1):
        network = []
        # Each nested array represents the weights of all the neurons from 
        # the previous layer (including the bias) to one neuron from the current layer.

        # hidden_layer1 = [[random() for i in range(number_of_inputs + 1)] for i in range(number_of_hidden_neurons)]
        # hidden_layer2 = [[random() for i in range(number_of_hidden_neurons + 1)] for i in range(number_of_hidden_neurons)]
        # output_layer = [[random() for i in range(number_of_hidden_neurons + 1)] for i in range(number_of_outputs)]
        
        hidden_layer1 = np.random.rand(number_of_hidden_neurons, number_of_inputs + 1)
        hidden_layer2 = np.random.rand(number_of_hidden_neurons, number_of_hidden_neurons + 1)
        output_layer = np.random.rand(number_of_outputs, number_of_hidden_neurons + 1)
        network.append(hidden_layer1)
        network.append(hidden_layer2)
        network.append(output_layer)
        return network
    
    def compute_forward_pass(self):
        pass

    def train(self, network):
        pass

    def predict(self):
        pass

def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    bank_train_path = os.path.join(script_directory, '..', 'Datasets', 'bank-note', 'train.csv')
    bank_test_path = os.path.join(script_directory, '..', 'Datasets', 'bank-note', 'test.csv')

    nn = Neural_Network()

    # Using bank dataset
        # Upload training dataset
    train_dataset = pd.read_csv(bank_train_path, header=None)
    train_dataset.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
        # Upload testing dataset
    test_dataset = pd.read_csv(bank_test_path, header=None)
    test_dataset.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
        # Create copies of training and testing datasets for predicting
    train_predicted_dataset = pd.DataFrame(train_dataset)
    train_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
    test_predicted_dataset = pd.DataFrame(test_dataset)
    test_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
        # Create training and test arrays
    train_inputs = train_dataset.drop('label', axis=1).to_numpy()
    train_labels = train_dataset['label'].to_numpy().copy()
    train_labels[train_labels == 0.0] = -1.0
    test_inputs = test_dataset.drop('label', axis=1).to_numpy()
    test_labels = test_dataset['label'].to_numpy().copy()
    test_labels[test_labels == 0.0] = -1.0

if __name__ == "__main__":
    main()