import os
import numpy as np
import pandas as pd

class Neural_Network:
    def create_network(self, number_of_inputs=4, number_of_hidden_neurons=4, number_of_outputs=1):
        # Each layer has nested dictionaries representing the neurons in that layer.
        # Each dictionary contains an array with the weights of all the neurons from 
        # the previous layer (including the bias) to one neuron from the current layer.
        network = []
        hidden_layer1 = [{'weights': np.random.rand(number_of_inputs + 1)} for i in range(number_of_hidden_neurons)]
        hidden_layer2 = [{'weights': np.random.rand(number_of_hidden_neurons + 1)} for i in range(number_of_hidden_neurons)]
        output_layer = [{'weights': np.random.rand(number_of_hidden_neurons + 1)} for i in range(number_of_outputs)]
        network.append(hidden_layer1)
        network.append(hidden_layer2)
        network.append(output_layer)
        return network
    
    def compute_forward_pass(self, network, row):
        inputs = row
        # Add bias to x
        row = np.append(row, 1)
        for layer_index, layer in enumerate(network):
            new_inputs = []
            for neuron in layer:
                activation = np.dot(neuron['weights'], row)
                if layer_index != len(network) - 1:
                    # Using the sigmoid activation function for hidden layers
                    output = 1.0 / (1.0 + np.exp(-activation))
                else:
                    # Using only the linear combination for output layer
                    output = activation
                neuron['output'] = output
                new_inputs.append(output)
            inputs = new_inputs
        # Return the output from the last layer
        return inputs
    
    def compute_backpropagation(self, network, actual_label):
        for layer_index in reversed(range(len(network))):
            layer = network[layer_index]
        # for layer_index, layer in enumerate(reversed(range(len(network)))):
        # for layer_index, layer in enumerate(reversed(network)):
            errors = []
            # For all the layers except the output layer
            if layer_index != len(network) - 1:
                for neuron_index in range(len(layer)):
                    error = 0.0
                    for previous_neuron_index in range(len(network[layer_index + 1])):
                        # Add up loss from previous layer
                        error += network[layer_index + 1][previous_neuron_index] * losses[layer_index + 1][previous_neuron_index]
                        errors.append(error)
            # For the output layer only
            else:
                for neuron_index, neuron in enumerate(layer):
                    predicted_label = outputs[layer_index][neuron_index]
                    errors.append(predicted_label - actual_label)
            for neuron_index, neuron in enumerate(layer):
                neuron_output = outputs[layer_index][neuron_index]
                # Compute loss using the derivative of the sigmoid function
                losses[layer_index][neuron_index] = errors[neuron_index] * neuron_output * (1.0 - neuron_output)

    # def create_network(self, number_of_inputs=4, number_of_hidden_neurons=4, number_of_outputs=1):
    #     # Each layer has nested arrays representing the neurons in that layer.
    #     # Each nested array contains the weights of all the neurons from 
    #     # the previous layer (including the bias) to one neuron from the current layer.
    #     network = []
    #     hidden_layer1 = np.random.rand(number_of_hidden_neurons, number_of_inputs + 1)
    #     hidden_layer2 = np.random.rand(number_of_hidden_neurons, number_of_hidden_neurons + 1)
    #     output_layer = np.random.rand(number_of_outputs, number_of_hidden_neurons + 1)
    #     network.append(hidden_layer1)
    #     network.append(hidden_layer2)
    #     network.append(output_layer)

    #     outputs = []
    #     hidden_layer1_outputs = np.zeros(number_of_hidden_neurons)
    #     hidden_layer2_outputs = np.zeros(number_of_hidden_neurons)
    #     output_layer_outputs = np.zeros(number_of_outputs)
    #     outputs.append(hidden_layer1_outputs)
    #     outputs.append(hidden_layer2_outputs)
    #     outputs.append(output_layer_outputs)

    #     losses = []
    #     hidden_layer1_losses = np.zeros(number_of_hidden_neurons)
    #     hidden_layer2_losses = np.zeros(number_of_hidden_neurons)
    #     output_layer_losses = np.zeros(number_of_outputs)
    #     losses.append(hidden_layer1_losses)
    #     losses.append(hidden_layer2_losses)
    #     losses.append(output_layer_losses)

    #     return network, outputs, losses
    
    # def compute_forward_pass(self, network, outputs, row):
    #     inputs = row
    #     # Add bias to x
    #     row = np.append(row, 1)
    #     for layer_index, layer in enumerate(network):
    #         new_inputs = []
    #         for neuron_index, neuron in enumerate(layer):
    #             activation = np.dot(neuron, row)
    #             if layer_index != len(network) - 1:
    #                 # Using the sigmoid activation function
    #                 output = 1.0 / (1.0 + np.exp(-activation))
    #             else:
    #                 # Using only linear combination for output layer
    #                 output = activation
    #             outputs[layer_index][neuron_index] = output
    #             new_inputs.append(output)
    #         inputs = new_inputs
    #     # Return the output from the last layer
    #     return inputs
    
    # def compute_backpropagation(self, network, outputs, losses, actual_label):
    #     for layer_index in reversed(range(len(network))):
    #         layer = network[layer_index]
    #     # for layer_index, layer in enumerate(reversed(range(len(network)))):
    #     # for layer_index, layer in enumerate(reversed(network)):
    #         errors = []
    #         # For all the layers except the output layer
    #         if layer_index != len(network) - 1:
    #             for neuron_index in range(len(layer)):
    #                 error = 0.0
    #                 for previous_neuron_index in range(len(network[layer_index + 1])):
    #                     # Add up loss from previous layer
    #                     error += network[layer_index + 1][previous_neuron_index] * losses[layer_index + 1][previous_neuron_index]
    #                     errors.append(error)
    #         # For the output layer only
    #         else:
    #             for neuron_index, neuron in enumerate(layer):
    #                 predicted_label = outputs[layer_index][neuron_index]
    #                 errors.append(predicted_label - actual_label)
    #         for neuron_index, neuron in enumerate(layer):
    #             neuron_output = outputs[layer_index][neuron_index]
    #             # Compute loss using the derivative of the sigmoid function
    #             losses[layer_index][neuron_index] = errors[neuron_index] * neuron_output * (1.0 - neuron_output)

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
    network = nn.create_network()
    final_output = nn.compute_forward_pass(network, row)
    # nn.compute_backpropagation(network, outputs, losses, train_dataset['label'].sample(n=1).iloc[0])

    print('The network is', network)
    # print()
    # print('The neurons outputs are', outputs)
    # print()
    # print('The neurons losses are', losses)
    # print()
    # print('The final output is', final_output)

if __name__ == "__main__":
    main()