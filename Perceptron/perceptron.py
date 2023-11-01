import os
import numpy as np
import pandas as pd

class Perceptron:
    def predict_row(row, weights):
        activation = weights[0]
        # From 0 to number of input features - 1
        for i in range(len(row) - 1):
            # Compute dot product
            activation = activation + weights[i + 1] * row[i]
        if activation >= 0.0:
            return 1
        else:
            return -1
        
def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    bank_train_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')
    bank_test_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'test.csv')

    perceptron = Perceptron()

    # Using bank dataset
        # Upload training dataset
    bank_train_dataset = pd.read_csv(bank_train_path, header=None)
    bank_train_dataset.columns = ['age','job','marital','education',
                                'default','balance','housing', 'loan', 
                                'contact', 'day', 'month', 'duration', 
                                'campaign', 'pdays', 'previous', 'poutcome', 'label']
    bank_features = {'age': [], 
                    'job': ['admin', 'unknown', 'unemployed', 'management', 
                            'housemaid', 'entrepreneur', 'student', 'blue-collar', 
                            'self-employed', 'retired', 'technician', 'services'], 
                    'marital': ['married', 'divorced', 'single'], 
                    'education': ['unknown', 'primary', 'secondary', 'tertiary'], 
                    'default': ['yes', 'no'], 
                    'balance': [], 
                    'housing': ['yes', 'no'], 
                    'loan': ['yes', 'no'], 
                    'contact': ['unknown', 'telephone', 'cellular'], 
                    'day': [], 
                    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                              'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                    'duration': [], 
                    'campaign': [],
                    'pdays': [], 
                    'previous': [],
                    'poutcome': ['unknown', 'other', 'failure', 'success']}
        # Upload testing dataset
    bank_test_dataset = pd.read_csv(bank_test_path, header=None)
    bank_test_dataset.columns = ['age','job','marital','education',
                                'default','balance','housing', 'loan', 
                                'contact', 'day', 'month', 'duration', 
                                'campaign', 'pdays', 'previous', 'poutcome', 'label']
        # Create copy of testing dataset for predicting
    bank_predicted_dataset = pd.DataFrame(bank_test_dataset)
    bank_predicted_dataset['label'] = ""   # or = np.nan for numerical columns

    # tennis_train_path = os.path.join(script_directory, '..', 'Datasets', 'tennis', 'train.csv')
    #  # Using tennis dataset
    #     # Upload training dataset
    # tennis_train_dataset = pd.read_csv(tennis_train_path, header=None)
    # tennis_train_dataset.columns = ['Outlook','Temp','Humidity','Wind','label']
    # tennis_features = {'Outlook': ['Sunny', 'Overcast', 'Rain'], 
    #                    'Temp': ['Hot', 'Medium', 'Cool'], 
    #                    'Humidity': ['High', 'Normal', 'Low'], 
    #                    'Wind': ['Strong', 'Weak']}
    #     # Upload testing dataset
    # tennis_test_dataset = pd.read_csv(tennis_train_path, header=None)
    # tennis_test_dataset.columns = ['Outlook','Temp','Humidity','Wind','label']
    #     # Create copy of testing dataset for predicting
    # tennis_predicted_dataset = pd.DataFrame(tennis_test_dataset)
    # tennis_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
    # tennis_tree = DT.ID3(tennis_train_dataset, tennis_features, 4, 6)
    # tennis_predicted_dataset = DT.predict(tennis_tree, tennis_predicted_dataset, tennis_features)
    # tennis_error = DT.prediction_error(tennis_test_dataset['label'].to_numpy(), tennis_predicted_dataset['label'].to_numpy())
    # DT.print_tree(tennis_tree)
    # print('The prediction error for this tree is', tennis_error)
    # tennis_predicted_dataset = RF.random_forest(tennis_train_dataset, tennis_predicted_dataset, tennis_features, 500, 4)
    # tennis_bagging_error = DT.prediction_error(tennis_test_dataset['label'].to_numpy(), tennis_predicted_dataset['label'].to_numpy())
    # print('The prediction error for this tree is', tennis_bagging_error)

if __name__ == "__main__":
    main()