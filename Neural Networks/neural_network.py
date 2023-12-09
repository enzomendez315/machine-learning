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
        # hidden_layer1 = [{'weights': np.zeros(number_of_inputs + 1)} for i in range(number_of_hidden_neurons)]
        # hidden_layer2 = [{'weights': np.zeros(number_of_hidden_neurons + 1)} for i in range(number_of_hidden_neurons)]
        # output_layer = [{'weights': np.zeros(number_of_hidden_neurons + 1)} for i in range(number_of_outputs)]
        network.append(hidden_layer1)
        network.append(hidden_layer2)
        network.append(output_layer)
        return network
    
    def compute_forward_pass(self, network, row):
        inputs = row
        # Add bias to x
        row = np.append(row, 1)
        for layer_index, layer in enumerate(network):
            outputs = []
            for neuron in layer:
                weights = neuron['weights']
                # Start with the bias
                activation = weights[-1]
                for i in range(len(weights) - 1):
                    # Add every input times its weight
                    activation += weights[i] * inputs[i]
                activation = np.clip(activation, 1e-7, 1-1e-7) #CHECK THIS LINE -----------
                if layer_index != len(network) - 1:
                    # Using the sigmoid activation function for hidden layers
                    output = 1.0 / (1.0 + np.exp(-activation))
                else:
                    # Using only the linear combination for output layer
                    output = activation
                neuron['output'] = output
                outputs.append(output)
            inputs = outputs
        # Return the output from the last layer
        return inputs[0]
    
    def compute_backpropagation(self, network, actual_label):
        # Start from output layer and work your way down
        for layer_index in reversed(range(len(network))):
            layer = network[layer_index]
            losses = []
            # For all the hidden layers
            if layer_index != len(network) - 1:
                for neuron_index in range(len(layer)):
                    loss = 0.0
                    for previous_neuron in network[layer_index + 1]:
                        # Add up the loss from previous layer: weight times loss of previous neuron
                        loss += previous_neuron['weights'][neuron_index] * previous_neuron['loss']
                        losses.append(loss)
            # For the output layer only
            else:
                for neuron_index, neuron in enumerate(layer):
                    predicted_label = neuron['output']
                    losses.append(predicted_label - actual_label)
            for neuron_index, neuron in enumerate(layer):
                neuron_output = neuron['output']
                if layer_index != len(network) - 1:
                    # Compute loss using the derivative of the sigmoid function for hidden layers
                    neuron['loss'] = losses[neuron_index] * neuron_output * (1.0 - neuron_output)
                else:
                    # Compute loss using the derivative of MSE for output layer
                    neuron['loss'] = losses[neuron_index]

    def train(self, network, train_dataset, learning_rate, epochs=100):
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                actual_label = dataset_row['label']
                if actual_label == 0.0:
                    actual_label = -1.0
                row = dataset_row.tolist()[:-1]
                self.compute_forward_pass(network, row)
                self.compute_backpropagation(network, actual_label)
                gamma = next(learning_rate)
                for layer_index, layer in enumerate(network):
                    inputs = row
                    np.append(inputs, 1)
                    if layer_index != 0:
                        inputs = [neuron['output'] for neuron in network[layer_index - 1]]
                    for neuron in layer:
                        for output_index in range(len(inputs)):
                            # Update the weights
                            neuron['weights'][output_index] -= gamma * neuron['loss'] * inputs[output_index]
                        # Update the bias
                        neuron['weights'][-1] -= gamma * neuron['loss']

    def predict(self, network, dataset):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()[:-1]
            predicted_label = self.compute_forward_pass(network, row)
            if np.sign(predicted_label) == -1.0:
                predicted_label = 0
            else:
                predicted_label = 1
            dataset.at[index, 'label'] = predicted_label
        return dataset
    
    def compute_error(self, actual_labels, predicted_labels):
        incorrect_examples = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] != predicted_labels[i]:
                incorrect_examples += 1
        return incorrect_examples / len(actual_labels)

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

    def learning_rate(initial_gamma, alpha):
        t = 0
        while True:
            yield initial_gamma / (1 + (initial_gamma / alpha) * t)
            t += 1

    width_values = [5, 10, 25, 50, 100]

    for width in width_values:
        network = nn.create_network(number_of_hidden_neurons=width)
        nn.train(network, train_dataset, learning_rate(0.1, 1))
        train_predicted_dataset = nn.predict(network, train_predicted_dataset)
        test_predicted_dataset = nn.predict(network, test_predicted_dataset)
        train_error = nn.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
        test_error = nn.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
        print('The training error for width =', width, 'is', train_error)
        print('The test error for width =', width, 'is', test_error)

    # network = nn.create_network(number_of_hidden_neurons=6)
    # nn.train(network, train_dataset, learning_rate(0.1, 1))
    # train_predicted_dataset = nn.predict(network, train_predicted_dataset)
    # test_predicted_dataset = nn.predict(network, test_predicted_dataset)
    # train_error = nn.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
    # test_error = nn.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
    # print('The training error for width =', train_error)
    # print('The test error for width =', test_error)

if __name__ == "__main__":
    main()