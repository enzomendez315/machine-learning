import os
import numpy as np
import pandas as pd

class Neural_Network:
    def create_network(self, number_of_inputs=4, number_of_hidden_neurons=4, number_of_outputs=1):
        # Each layer has nested arrays representing the neurons in that layer.
        # Each nested array contains the weights of all the neurons from 
        # the previous layer (including the bias) to one neuron from the current layer.
        network = []
        hidden_layer1 = np.random.rand(number_of_hidden_neurons, number_of_inputs + 1)
        hidden_layer2 = np.random.rand(number_of_hidden_neurons, number_of_hidden_neurons + 1)
        output_layer = np.random.rand(number_of_outputs, number_of_hidden_neurons + 1)
        network.append(hidden_layer1)
        network.append(hidden_layer2)
        network.append(output_layer)

        outputs = []
        hidden_layer1_outputs = np.zeros(number_of_hidden_neurons)
        hidden_layer2_outputs = np.zeros(number_of_hidden_neurons)
        output_layer_outputs = np.zeros(number_of_outputs)
        outputs.append(hidden_layer1_outputs)
        outputs.append(hidden_layer2_outputs)
        outputs.append(output_layer_outputs)

        return network, outputs
    
    def compute_forward_pass(self, network, outputs, row):
        inputs = row
        # Add bias to x
        row = np.append(row, 1)
        for layer_index, layer in enumerate(network):
            new_inputs = []
            for neuron_index, neuron in enumerate(layer):
                activation = np.dot(neuron, row)
                if layer_index != len(network) - 1:
                    # Using the sigmoid activation function
                    output = 1.0 / (1.0 + np.exp(-activation))
                else:
                    # Using linear combination for output layer
                    output = activation
                outputs[layer_index][neuron_index] = output
                new_inputs.append(output)
            inputs = new_inputs
        # Return the output from the last layer
        return inputs
    
    def compute_backpropagation(self, network):
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

    def learning_rate(initial_gamma, alpha):
        t = 0
        while True:
            yield initial_gamma / (1 + (initial_gamma / alpha) * t)
            t += 1

    row = train_dataset.drop('label', axis=1).sample(n=1).to_numpy()
    network, outputs = nn.create_network()
    inputs = nn.compute_forward_pass(network, outputs, row)

    print(network)
    print()
    print(outputs)
    print()
    print(inputs)

if __name__ == "__main__":
    main()