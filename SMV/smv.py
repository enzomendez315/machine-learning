import os
import numpy as np
import pandas as pd
from copy import deepcopy

class SMV:
    def train_standard(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Update weights
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
        return weights
    
    def train_voted(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        weight_vectors = []
        votes = []
        vote_count = 0
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Add weight vector and its vote
                        weight_vectors.append(deepcopy(weights))
                        votes.append(vote_count)
                        # Create new weight vector
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
                        vote_count = 1
                else:
                    vote_count += 1
        return weight_vectors, votes
    
    def train_averaged(self, train_dataset, epochs, learning_rate):
        # Initialize weights. Bias is the first element
        weights = [0.0 for i in range(len(train_dataset.columns))]
        averages = [0.0 for i in range(len(train_dataset.columns))]
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = dataset_row.tolist()
                prediction = self._predict_row(row, weights)
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                if prediction != actual_label:
                    weights[0] += learning_rate * actual_label
                    for i in range(len(row) - 1):
                        # Update weights
                        weights[i+1] = weights[i+1] + learning_rate * actual_label * row[i]
                else:
                    for i in range(len(weights)):
                        averages[i] += weights[i]
        return averages

    def _predict_row(self, row, weights):
        # The bias is the first element of weights vector
        activation = weights[0]
        for i in range(len(row) - 1):
            # Compute the dot product for the rest of the elements
            activation = activation + weights[i+1] * row[i]
        if activation >= 0.0:
            return 1
        else:
            return -1
        
    def predict_standard(self, dataset, weights):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            prediction = self._predict_row(row, weights)
            if prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset

    def predict_voted(self, dataset, weights, votes):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            voted_prediction = 0
            for i in range(len(weights)):
                prediction = 0
                for j in range(len(row) - 1):
                    prediction = prediction + weights[i][j+1] * row[j]
                if prediction >= 0.0:
                    prediction = 1
                else:
                    prediction = -1
                voted_prediction += votes[i] * prediction
            if voted_prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset

    def predict_averaged(self, dataset, weights):
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()
            prediction = self._predict_row(row, weights)
            if prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
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

    smv = SMV()

    # Using bank dataset
        # Upload training dataset
    train_dataset = pd.read_csv(bank_train_path, header=None)
    train_dataset.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
        # Upload testing dataset
    test_dataset = pd.read_csv(bank_test_path, header=None)
    test_dataset.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
        # Create copies of testing dataset for predicting
    standard_predicted_dataset = pd.DataFrame(test_dataset)
    standard_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
    voted_predicted_dataset = pd.DataFrame(standard_predicted_dataset)
    averaged_predicted_dataset = pd.DataFrame(standard_predicted_dataset)

    # Results using standard perceptron
    standard_weights = perceptron.train_standard(train_dataset, 10, 0.5)
    standard_predicted_dataset = perceptron.predict_standard(standard_predicted_dataset, standard_weights)
    standard_error = perceptron.compute_error(test_dataset['label'].to_numpy(), standard_predicted_dataset['label'].to_numpy())
    print('The weights using the standard perceptron are', standard_weights)
    print('The standard perceptron error is', standard_error)

    # Results using voted perceptron
    voted_weights, votes = perceptron.train_voted(train_dataset, 10, 0.5)
    voted_predicted_dataset = perceptron.predict_voted(voted_predicted_dataset, voted_weights, votes)
    voted_error = perceptron.compute_error(test_dataset['label'].to_numpy(), voted_predicted_dataset['label'].to_numpy())
    #print('The weights using the voted perceptron are', voted_weights)
    for i in range(len(voted_weights)):
        print(voted_weights[i])
    print('The votes for the weight vectors using the voted perceptron are', votes)
    print('The voted perceptron error is', voted_error)

    # Results using averaged perceptron
    averaged_weights = perceptron.train_averaged(train_dataset, 10, 0.5)
    averaged_predicted_dataset = perceptron.predict_averaged(averaged_predicted_dataset, averaged_weights)
    averaged_error = perceptron.compute_error(test_dataset['label'].to_numpy(), averaged_predicted_dataset['label'].to_numpy())
    print('The weights using the averaged perceptron are', averaged_weights)
    print('The averaged perceptron error is', averaged_error)

if __name__ == "__main__":
    main()