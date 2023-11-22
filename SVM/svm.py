import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds

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
    
    def train_dual(self, train_dataset, C, kernel, gammas=None):
        labels = train_dataset['label'].to_numpy().copy()
        labels[labels == 0.0] = -1.0
        #labels = np.asmatrix(labels.reshape((872, 1)))
        inputs = train_dataset.drop('label', axis=1).to_numpy()
        #inputs = np.asmatrix(inputs)
        N = train_dataset.shape[0]
        initial_x = np.zeros(N)
        # Create bound from 0 to C
        _bounds = Bounds(np.full((N), 0), np.full((N), C))
        _constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, labels)}
        def dual_objective(alpha):
            # Find the objective function that is being maximized
            kernel_labels = labels.T * kernel(inputs, inputs.T) * labels
            sum = alpha.T.dot(kernel_labels[0,0] * alpha)
            return 1/2 * sum - np.sum(alpha)
        alphas = minimize(dual_objective, initial_x, method='SLSQP', bounds=_bounds, constraints=_constraints)
        alphas = alphas.x
        # Set values to be 0 or C
        alphas[np.isclose(alphas, 0)] = 0
        alphas[np.isclose(alphas, C)] = C
        return alphas

    def recover_dual_weights(self, alphas, dataset):
        # Separate the labels
        labels = dataset['label'].to_numpy().copy()
        labels[labels == 0.0] = -1.0
        labels = np.asmatrix(labels.reshape((872, 1)))
        # Separate the inputs
        inputs = dataset.drop('label', axis=1).to_numpy()
        inputs = np.asmatrix(inputs)
        return np.sum((alphas * labels)[0,0] * inputs, axis=0)
    
        # # Initialize weights
        # weights = np.zeros(len(dataset.columns) - 1)
        # for i in range(alphas.size):
        #     weights += inputs[i] * alphas[i] * labels[i]
        # return weights

    def recover_dual_bias(self, alphas, dataset):
        # Separate the labels
        labels = dataset['label'].to_numpy().copy()
        labels[labels == 0.0] = -1.0
        labels = np.asmatrix(labels.reshape((labels.size, 1)))
        # Separate the inputs
        inputs = dataset.drop('label', axis=1).to_numpy()
        inputs = np.asmatrix(inputs)
        return np.mean(labels - np.sum((alphas * labels)[0,0] * np.dot(inputs, inputs.T)))
    
        # labels = dataset['label'].to_numpy()
        # labels[labels == 0.0] = -1.0
        # inputs = dataset.drop('label', axis=1).to_numpy()
        # bias = 0
        # num_nonzero = 0
        # for i in range(len(alphas)):
        #     if alphas[i] > 1e-11:
        #         inner_sum = 0
        #         for j in range(len(alphas)):
        #             inner_sum += alphas[j] * labels[j] * np.dot(inputs[i], inputs[j])
        #         bias += labels[i] - inner_sum
        #         num_nonzero += 1
        # return bias / num_nonzero

    def predict_dual(self, dataset, inputs, labels, alphas, bias, kernel):
        labels = np.asmatrix(labels.reshape((labels.size, 1)))
        inputs = np.asmatrix(inputs)
        prediction = np.sign(np.sum((alphas * labels)[0, 0] * kernel(inputs, inputs.T) + bias, axis=1))
        for i in range(len(prediction)):
            if prediction[i] >= 0.0:
                dataset.at[i, 'label'] = 1
            else:
                dataset.at[i, 'label'] = 0
        return dataset

        # for index, dataset_row in dataset.iterrows():
        #     row = dataset_row.tolist()[:-1]
        #     row = np.array(row)
        #     # Add bias to x
        #     row = np.append(row, 1)
        #     prediction = np.dot(weights, row)
        #     if prediction >= 0.0:
        #         dataset.at[index, 'label'] = 1
        #     else:
        #         dataset.at[index, 'label'] = 0
        # return dataset
    
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
        # Create training and test arrays
    train_inputs = train_dataset.drop('label', axis=1).to_numpy()
    train_labels = train_dataset['label'].to_numpy().copy()
    train_labels[train_labels == 0.0] = -1.0
    test_inputs = test_dataset.drop('label', axis=1).to_numpy()
    test_labels = test_dataset['label'].to_numpy().copy()
    test_labels[test_labels == 0.0] = -1.0

    C_values = [100/873, 500/873, 700/873]
    gamma_values = [0.1, 0.5, 1, 5, 100]

    def learning_rate_A(initial_gamma, alpha):
        t = 0
        while True:
            yield initial_gamma / (1 + (initial_gamma/alpha) * t)
            t += 1
    
    def learning_rate_B(initial_gamma):
        t = 0
        while True:
            yield initial_gamma / (1 + t)
            t += 1

    # # Results using primal svm
    # for C in C_values:
    #     primal_weights = svm.train_primal(train_dataset, 100, learning_rate_A(0.1, 1), C)
    #     train_predicted_dataset = svm.predict_primal(train_predicted_dataset, primal_weights)
    #     test_predicted_dataset = svm.predict_primal(test_predicted_dataset, primal_weights)
    #     train_error = svm.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
    #     test_error = svm.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
    #     print('The training error for C =', C, 'in the primal domain is', train_error)
    #     print('The test error for C =', C, 'in the primal domain is', test_error)

    # for C in C_values:
    #     primal_weights = svm.train_primal(train_dataset, 100, learning_rate_B(0.1), C)
    #     train_predicted_dataset = svm.predict_primal(train_predicted_dataset, primal_weights)
    #     test_predicted_dataset = svm.predict_primal(test_predicted_dataset, primal_weights)
    #     train_error = svm.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
    #     test_error = svm.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
    #     print('The training error for C =', C, 'in the primal domain is', train_error)
    #     print('The test error for C =', C, 'in the primal domain is', test_error)

    # Results using dual svm
    for C in C_values:
        dual_alphas = svm.train_dual(train_dataset, C, np.dot)
        dual_weights = svm.recover_dual_weights(dual_alphas, train_dataset)
        dual_bias = svm.recover_dual_bias(dual_alphas, train_dataset)
        train_predicted_dataset = svm.predict_dual(train_predicted_dataset, train_inputs, train_labels, dual_alphas, dual_bias, np.dot)
        #test_predicted_dataset = svm.predict_dual(test_predicted_dataset, test_inputs, test_labels, dual_alphas, dual_bias, np.dot)
        train_error = svm.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
        #test_error = svm.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
        print('The training error for C =', C, 'in the dual domain is', train_error)
        #print('The test error for C =', C, 'in the dual domain is', test_error)


        # dual_weights = np.hstack([dual_weights, np.full((dual_weights.shape[0], 1), dual_bias)])
        # train_predicted_dataset = svm.predict_primal(train_predicted_dataset, dual_weights)
        # test_predicted_dataset = svm.predict_primal(test_predicted_dataset, dual_weights)
        # train_error = svm.compute_error(train_dataset['label'].to_numpy(), train_predicted_dataset['label'].to_numpy())
        # test_error = svm.compute_error(test_dataset['label'].to_numpy(), test_predicted_dataset['label'].to_numpy())
        # print('The training error for C =', C, 'in the dual domain is', train_error)
        # print('The test error for C =', C, 'in the dual domain is', test_error)


if __name__ == "__main__":
    main()