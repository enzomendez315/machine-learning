import os
import numpy as np
import pandas as pd

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
        DT = DecisionTree()
        stump = DT.decision_stump(data, features, numerical_features)

def predict(X):
    pass

class DecisionTree:
    def entropy(self, data):
        total_size = data.shape[0]
        entropy = 0
        labels = data['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = data[data['label'] == value].shape[0]
            if subset_size == 0:
                continue
            entropy -= (subset_size / total_size) * np.log2(subset_size / total_size)
        return entropy

    def majority_error(self, data):
        total_size = data.shape[0]
        majority_error = 1
        labels = data['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = data[data['label'] == value].shape[0]
            subset_ratio = subset_size / total_size
            if subset_ratio < majority_error:
                majority_error = subset_ratio
        return majority_error

    def gini_index(self, data):
        total_size = data.shape[0]
        gini_index = 1
        labels = data['label'].unique()
        for value in labels:
            # Create a subset of all rows that have the same feature value
            subset_size = data[data['label'] == value].shape[0]
            gini_index -= (subset_size / total_size) ** 2
        return gini_index

    def gain(self, feature, data, purity_measure):
        total_size = data.shape[0]
        weighted_average = 0
        features = data[feature].unique()
        subset_purity = purity_measure(data)
        for value in features:
            # Create a subset of all rows that have the same feature value
            feature_value_subset = data[data[feature] == value]
            subset_size = feature_value_subset.shape[0]
            feature_value_purity = purity_measure(feature_value_subset)
            weighted_average += (subset_size / total_size) * feature_value_purity
        return subset_purity - weighted_average

    def feature_for_split(self, data, purity_measure):
        # Remove label column
        features = data.drop('label', axis=1)
        # Set to -1 for the case that gain = 0 for some feature
        highest_gain = -1
        purest_feature = None
        for feature in features:
            feature_gain = self.gain(feature, data, purity_measure)
            if highest_gain < feature_gain:
                highest_gain = feature_gain
                purest_feature = feature
        return purest_feature

    def ID3_entropy(self, data, features, depth):
        # All examples have the same label
        if len(data['label'].unique()) == 1:
            # Return a leaf node with that label
            return Node(None, None, data['label'].iloc[0])
        
        # Features are empty or depth is reached
        if len(features) == 0 or depth == 0:
            # Return a leaf node with the most common label
            label_count = data['label'].value_counts()
            return Node(None, None, label_count.idxmax())

        root = Node(None, None, None)
        purest_feature = self.feature_for_split(data, self.entropy)
        root.feature = purest_feature
        purest_feature_values = data[purest_feature].unique()
        root.values = {}

        for value in purest_feature_values:
            # Add a new tree branch for every value
            root.values[value] = None

            # Create a subset Sv of examples in S where A = v
            subset = data[data[purest_feature] == value]

            # S_v is empty
            if subset.shape[0] == 0:
                # Add a leaf node with the most common label in S
                most_common_label = data['label'].value_counts().idxmax()
                root.values[value] = Node(None, None, most_common_label)
            else:
                # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                if purest_feature in features.columns:
                    features = features.drop(purest_feature, axis=1)
                subtree_node = self.ID3_entropy(subset, features, depth - 1)
                root.values[value] = subtree_node
        return root
    
    def ID3_entropy_numerical(self, data, features, numerical_features, depth):
        # All examples have the same label
        if len(data['label'].unique()) == 1:
            # Return a leaf node with that label
            return Node(None, None, data['label'].iloc[0])
        
        # Features are empty or depth is reached
        if len(features) == 0 or depth == 0:
            # Return a leaf node with the most common label
            label_count = data['label'].value_counts()
            return Node(None, None, label_count.idxmax())

        root = Node(None, None, None)
        purest_feature = self.feature_for_split(data, self.entropy)
        root.feature = purest_feature
        root.values = {}

        # Check if the purest feature is numerical. Split accordingly.
        if purest_feature in numerical_features:
            # Convert to binary feature.
            median_value = data[purest_feature].median()
            purest_feature_values = {'more than or equal to ' + str(median_value), 'less than ' + str(median_value)}
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                if 'more than' in value:
                    subset = data[data[purest_feature] >= median_value]
                else:
                    subset = data[data[purest_feature] < median_value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = data['label'].value_counts().idxmax()
                    root.values[value] = Node(None, None, most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features.columns:
                        features = features.drop(purest_feature, axis=1)
                    subtree_node = self.ID3_entropy_numerical(subset, features, numerical_features, depth - 1)
                    root.values[value] = subtree_node
            return root
        
        else:
            purest_feature_values = data[purest_feature].unique()
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                subset = data[data[purest_feature] == value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = data['label'].value_counts().idxmax()
                    root.values[value] = Node(None, None, most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features.columns:
                        features = features.drop(purest_feature, axis=1)
                    subtree_node = self.ID3_entropy_numerical(subset, features, numerical_features, depth - 1)
                    root.values[value] = subtree_node
            return root
    
    def decision_stump(self, data, features, numerical_features, depth=1):
        # All examples have the same label
        if len(data['label'].unique()) == 1:
            # Return a leaf node with that label
            return Node(None, None, data['label'].iloc[0])
        
        # Features are empty or depth is reached
        if len(features) == 0 or depth == 0:
            # Return a leaf node with the most common label
            label_count = data['label'].value_counts()
            return Node(None, None, label_count.idxmax())

        root = Node(None, None, None)
        purest_feature = self.feature_for_split(data, self.entropy)
        root.feature = purest_feature
        root.values = {}

        # Check if the purest feature is numerical. Split accordingly.
        if purest_feature in numerical_features:
            # Convert to binary feature.
            median_value = data[purest_feature].median()
            purest_feature_values = {'more than or equal to ' + str(median_value), 'less than ' + str(median_value)}
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                if 'more than' in value:
                    subset = data[data[purest_feature] >= median_value]
                else:
                    subset = data[data[purest_feature] < median_value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = data['label'].value_counts().idxmax()
                    root.values[value] = Node(None, None, most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features.columns:
                        features = features.drop(purest_feature, axis=1)
                    subtree_node = self.decision_stump(subset, features, numerical_features, depth - 1)
                    root.values[value] = subtree_node
            return root
        
        else:
            purest_feature_values = data[purest_feature].unique()
            for value in purest_feature_values:
                # Add a new tree branch for every value
                root.values[value] = None

                # Create a subset Sv of examples in S where A = v
                subset = data[data[purest_feature] == value]

                # S_v is empty
                if subset.shape[0] == 0:
                    # Add a leaf node with the most common label in S
                    most_common_label = data['label'].value_counts().idxmax()
                    root.values[value] = Node(None, None, most_common_label)
                else:
                    # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
                    if purest_feature in features.columns:
                        features = features.drop(purest_feature, axis=1)
                    subtree_node = self.decision_stump(subset, features, numerical_features, depth - 1)
                    root.values[value] = subtree_node
            return root

    def print_tree(self, tree, indent=0):
        if not tree.values:
            print(" " * indent + tree.label)
            return
        
        print(" " * indent + f"{tree.feature}:")
        next_indent = indent + 4

        for feature_value, node in tree.values.items():
            print(" " * next_indent + f"{feature_value}:")
            self.print_tree(node, next_indent + 4)

class Node:
    def __init__(self, feature, values, label):
        self.feature = feature
        self.values = values    # values = {'value': Node}
        self.label = label

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
    DT = DecisionTree()
    bank_stump = DT.decision_stump(bank_dataset, features, numerical_features)
    DT.print_tree(bank_stump)

if __name__ == "__main__":
    main()