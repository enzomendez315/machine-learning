import os
import numpy as np
import pandas as pd
import random
import time

class DecisionTree:
    def entropy(self, dataset):
        total_size = dataset.shape[0]
        entropy = 0
        labels = dataset['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = dataset[dataset['label'] == value].shape[0]
            if subset_size == 0:
                continue
            entropy -= (subset_size / total_size) * np.log2(subset_size / total_size)
        return entropy

    def majority_error(self, dataset):
        total_size = dataset.shape[0]
        majority_error = 1
        labels = dataset['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = dataset[dataset['label'] == value].shape[0]
            subset_ratio = subset_size / total_size
            if subset_ratio < majority_error:
                majority_error = subset_ratio
        return majority_error

    def gini_index(self, dataset):
        total_size = dataset.shape[0]
        gini_index = 1
        labels = dataset['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = dataset[dataset['label'] == value].shape[0]
            gini_index -= (subset_size / total_size) ** 2
        return gini_index

    def gain(self, feature, dataset, purity_measure):
        total_size = dataset.shape[0]
        weighted_average = 0
        feature_values = dataset[feature].unique()
        subset_purity = purity_measure(dataset)
        for value in feature_values:
            # Create a subset of all rows that have the same feature value
            feature_value_subset = dataset[dataset[feature] == value]
            subset_size = feature_value_subset.shape[0]
            feature_value_purity = purity_measure(feature_value_subset)
            weighted_average += (subset_size / total_size) * feature_value_purity
        return subset_purity - weighted_average

    def feature_for_split(self, dataset, features, purity_measure):
        # Set to -1 for the case that gain = 0 for some feature
        highest_gain = -1
        purest_feature = None
        for feature in features:
            feature_gain = self.gain(feature, dataset, purity_measure)
            if highest_gain < feature_gain:
                highest_gain = feature_gain
                purest_feature = feature
        return purest_feature
    
    def ID3(self, dataset, features, depth):
        # All examples have the same label
        if len(dataset['label'].unique()) == 1:
            # Return a leaf node with that label
            return Node(label=dataset['label'].iloc[0])
        
        # Features are empty or depth is reached
        if len(features) == 0 or depth == 0:
            # Return a leaf node with the most common label
            label_count = dataset['label'].value_counts()
            return Node(label=label_count.idxmax())

        root = Node()
        purest_feature = self.feature_for_split(dataset, features, self.entropy)
        root.feature = purest_feature
        root.values = {}

        # Check if the purest feature is numerical. Split accordingly.
        if not features[purest_feature]:
            # Convert to binary feature.
            median_value = dataset[purest_feature].median()
            root.median = median_value
            purest_feature_values = {'more than or equal to ' + str(median_value), 'less than ' + str(median_value)}
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                if 'more than' in value:
                    subset = dataset[dataset[purest_feature] >= median_value]
                else:
                    subset = dataset[dataset[purest_feature] < median_value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = dataset['label'].value_counts().idxmax()
                    root.values[value] = Node(label=most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features:
                        new_features = {key: value for key, value in features.items() if key != purest_feature}
                    subtree_node = self.ID3(subset, new_features, depth - 1)
                    root.values[value] = subtree_node
            return root
        
        else:
            purest_feature_values = features[purest_feature]
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                subset = dataset[dataset[purest_feature] == value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = dataset['label'].value_counts().idxmax()
                    root.values[value] = Node(label=most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features:
                        new_features = {key: value for key, value in features.items() if key != purest_feature}
                    subtree_node = self.ID3(subset, new_features, depth - 1)
                    root.values[value] = subtree_node
            return root
    
    def predict(self, tree, dataset, feature_list):
        features = dataset.drop('label', axis=1)
        for index, row in features.iterrows():
            row_values = dict(row)
            predicted_label = self._predict_label(tree, row_values, feature_list)
            # Insert label at the right location
            dataset.at[index, 'label'] = predicted_label
        return dataset
    
    def _predict_label(self, tree, row_values, feature_list):
        predicted_label = ''

        # Leaf node indicates there is a label
        if not tree.values:
            return tree.label
        
        # Get the feature value using the feature found in tree
        feature_value = row_values[tree.feature]

        # Handle numerical values appropriately
        if not feature_list[tree.feature]:
            median_value = tree.median
            if feature_value >= median_value:
                subtree = tree.values['more than or equal to ' + str(median_value)]
                predicted_label = self._predict_label(subtree, row_values, feature_list)
                return predicted_label
            else:
                subtree = tree.values['less than ' + str(median_value)]
                predicted_label = self._predict_label(subtree, row_values, feature_list)
                return predicted_label
            
        if feature_value not in tree.values:
            feature_value = random.choice(list(tree.values.keys()))
        subtree = tree.values[feature_value]
        predicted_label = self._predict_label(subtree, row_values, feature_list)
        return predicted_label

    def prediction_error(self, actual_labels, predicted_labels):
        counter = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] != predicted_labels[i]:
                counter = counter + 1
        return counter / len(actual_labels)
    
    def dataset_sample(self, dataset, ratio=0.5):
        random_indices = dataset.sample(frac=ratio, replace=True).index
        return dataset.loc[random_indices]

    def print_tree(self, tree, indent=0):
        if not tree.values:
            print(" " * indent + tree.label)
            return
        
        print(" " * indent + f"{tree.feature}:")
        next_indent = indent + 4

        for feature_value, subtree in tree.values.items():
            print(" " * next_indent + f"{feature_value}:")
            self.print_tree(subtree, next_indent + 4)

class Node:
    def __init__(self, feature=None, values=None, label=None, median=None):
        self.feature = feature
        self.values = values    # values = {'value': Node}
        self.label = label
        self.median = median

class BaggedTrees:    
    def bagging(self, train_dataset, test_dataset, features, number_of_trees):
        DT = DecisionTree()
        predictions = []

        # Construct the trees and store their predictions
        for _ in range(number_of_trees):
            random_sample = DT.dataset_sample(train_dataset)
            tree = DT.ID3(random_sample, features, 20)
            prediction = DT.predict(tree, test_dataset, features)
            predictions.append(prediction['label'].to_numpy())
        
        # Update all the rows with the predictions
        for row_number in range(test_dataset.shape[0]):
            label = self._predict_label(predictions, row_number)
            test_dataset.loc[row_number, 'label'] = label
        return test_dataset
    
    def _predict_label(self, predictions, row_index):
        row_values = []

        for item in predictions:
            row_values.append(item[row_index])
        
        # Majority vote for classification or average for regression
        return max(row_values, key=row_values.count)

def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    bank_train_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')
    bank_test_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'test.csv')

    DT = DecisionTree()
    BT = BaggedTrees()

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

    # Vary the number of trees from 1 to 500, report how the training 
    # and test errors vary along with the tree number in a figure.
    for number_of_trees in range(1, 11):
        start_time = time.time()
        bank_predicted_dataset = BT.bagging(bank_train_dataset, bank_predicted_dataset, bank_features, number_of_trees)
        bank_training_error = DT.prediction_error(bank_train_dataset['label'].to_numpy(), bank_predicted_dataset['label'].to_numpy())
        bank_testing_error = DT.prediction_error(bank_test_dataset['label'].to_numpy(), bank_predicted_dataset['label'].to_numpy())
        end_time=time.time()
        total_time = end_time - start_time
        print('Bagging using', number_of_trees, 'trees:')
        print('Training error is', bank_training_error)
        print('Testing error is', bank_testing_error)
        print('Took', total_time, 'seconds')


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
    # tennis_predicted_dataset = BT.bagging(tennis_train_dataset, tennis_predicted_dataset, tennis_features, 10)
    # tennis_bagging_error = DT.prediction_error(tennis_test_dataset['label'].to_numpy(), tennis_predicted_dataset['label'].to_numpy())
    # print('The prediction error for this tree is', tennis_bagging_error)


if __name__ == "__main__":
    main()