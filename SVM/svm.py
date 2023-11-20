import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

class SVM:
    def train_primal(self, train_dataset, epochs, learning_rate, C=1.0):
        # Initialize weights
        weights = np.zeros(len(train_dataset.columns) - 1)
        # Add bias term at the end of weights vector
        weights = np.append(weights, 1)
        for epoch in range(epochs):
            # Shuffle the data
            train_dataset = train_dataset.sample(frac=1.0)
            for index, dataset_row in train_dataset.iterrows():
                row = np.array(dataset_row.tolist())
                # Set the correct label for calculation
                if row[-1] == 0.0:
                    actual_label = -1.0
                else:
                    actual_label = 1.0
                # Remove label and add bias to x
                row = np.append(row[:-1], 1)
                gamma = next(learning_rate)
                N = train_dataset.shape[0]
                prediction = actual_label * np.dot(weights, row)
                if prediction < 1:
                    weights = weights - gamma * np.append(weights[:-1], 0) + gamma * C * N * actual_label * row
                else:
                    weights[-1] = (1 - gamma) * weights[-1]
        return weights.tolist()
    
    def _predict_row(self, row, weights, actual_label):
        prediction = actual_label * np.dot(weights, row[:-1])
        return prediction
        
    def predict_primal(self, dataset, weights):
        # Bias is the last term of weights vector
        weights = np.array(weights)
        for index, dataset_row in dataset.iterrows():
            row = dataset_row.tolist()[:-1]
            row = np.array(row)
            # Add bias to x
            row = np.append(row, 1)
            prediction = np.dot(weights, row)
            if prediction >= 0.0:
                dataset.at[index, 'label'] = 1
            else:
                dataset.at[index, 'label'] = 0
        return dataset
    
    def hinge_loss(self, row):
        pass
    
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

    svm = SVM()

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

    C_values = [100/873, 500/873, 700/873]

    def learning_rate_A(initial_gamma, alpha):
        #yield initial_gamma
        t = 0
        while True:
            yield initial_gamma / (1 + (initial_gamma/alpha) * t)
            t += 1
    
    def learning_rate_B(initial_gamma):
        #yield initial_gamma
        t = 0
        while True:
            yield initial_gamma / (1 + t)
            t += 1

    # Results using primal svm
    for C in C_values:
        primal_weights = svm.train_primal(train_dataset, 100, learning_rate_A(0.1, 1), C)
        train_predicted_dataset = svm.predict_primal(train_predicted_dataset, primal_weights)
        test_predicted_dataset = svm.predict_primal(test_predicted_dataset, primal_weights)
        train_error = svm.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
        test_error = svm.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
        print('The training error for C =', C, 'in the primal domain is', train_error)
        print('The test error for C =', C, 'in the primal domain is', test_error)

    # # Results using standard perceptron
    # standard_weights = perceptron.train_standard(train_dataset, 10, 0.5)
    # standard_predicted_dataset = perceptron.predict_standard(standard_predicted_dataset, standard_weights)
    # standard_error = perceptron.compute_error(test_dataset['label'].to_numpy(), standard_predicted_dataset['label'].to_numpy())
    # print('The weights using the standard perceptron are', standard_weights)
    # print('The standard perceptron error is', standard_error)

    # # Results using voted perceptron
    # voted_weights, votes = perceptron.train_voted(train_dataset, 10, 0.5)
    # voted_predicted_dataset = perceptron.predict_voted(voted_predicted_dataset, voted_weights, votes)
    # voted_error = perceptron.compute_error(test_dataset['label'].to_numpy(), voted_predicted_dataset['label'].to_numpy())
    # #print('The weights using the voted perceptron are', voted_weights)
    # for i in range(len(voted_weights)):
    #     print(voted_weights[i])
    # print('The votes for the weight vectors using the voted perceptron are', votes)
    # print('The voted perceptron error is', voted_error)

    # # Results using averaged perceptron
    # averaged_weights = perceptron.train_averaged(train_dataset, 10, 0.5)
    # averaged_predicted_dataset = perceptron.predict_averaged(averaged_predicted_dataset, averaged_weights)
    # averaged_error = perceptron.compute_error(test_dataset['label'].to_numpy(), averaged_predicted_dataset['label'].to_numpy())
    # print('The weights using the averaged perceptron are', averaged_weights)
    # print('The averaged perceptron error is', averaged_error)

if __name__ == "__main__":
    main()