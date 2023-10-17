import os
import numpy as np
import pandas as pd
from ..DecisionTree import decision_tree as DT

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')
csv_debug_file_path = os.path.join(script_directory, '..', 'Datasets', 'tennis', 'train.csv')


def calculate_alpha(error):
    return (1/2) * np.log((1 - error) / error)

def update_weights(weight, alpha, prediction, target):
    return weight * np.exp(alpha * target * prediction)
    
def adaboost(data, features, numerical_features, number_classifiers):
    total_rows = data.shape[0]
    
    # Initialize weights
    weights = np.full(total_rows, (1 / total_rows))

    classifiers = []
    for i in range(number_classifiers):
        stump = DT.decision_stumps(data, features, numerical_features)

def predict(X):
    pass

def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    bank_file_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')

    bank_dataset = pd.read_csv(bank_file_path, header=None)
    bank_dataset.columns = ['age','job','marital','education',
                       'default','balance','housing', 'loan', 
                       'contact', 'day', 'month', 'duration', 
                       'campaign', 'pdays', 'previous', 'poutcome', 'label']
    features = bank_dataset.drop('label', axis=1)
    numerical_features = {'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'}
    bank_stump = DT.decision_stumps(bank_dataset, features, numerical_features)
    DT.print_tree(bank_stump)

if __name__ == "__main__":
    main()