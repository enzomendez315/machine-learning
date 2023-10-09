import os
import numpy as np
import pandas as pd

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_directory, 'car-4', 'train.csv')

def entropy(data):
    total_size = data.shape[0]
    entropy = 0
    labels = data['label'].unique()
    for value in labels:
        # Create a subset of all rows with label = value
        subset_size = data[data['label'] == value].shape[0]
        if subset_size == 0:
            continue
        entropy -= (subset_size / total_size) * np.log2(subset_size / total_size)
    return entropy

def majority_error(data):
    total_size = data.shape[0]
    majority_error = 1
    labels = data['label'].unique()
    for value in labels:
        # Create a subset of all rows with label = value
        subset_size = data[data['label'] == value].shape[0]
        subset_ratio = subset_size / total_size
        if subset_ratio < majority_error:
            majority_error = subset_ratio
    return majority_error

def gini_index(data):
    total_size = data.shape[0]
    gini_index = 1
    labels = data['label'].unique()
    for value in labels:
        # Create a subset of all rows with label = value
        subset_size = data[data['label'] == value].shape[0]
        gini_index -= (subset_size / total_size) ** 2
    return gini_index

def gain(feature, data, purity_measure):
    total_size = data.shape[0]
    weighted_average = 0
    features = data[feature].unique()
    subset_purity = purity_measure(data)
    for value in features:
        # Create a subset of all rows with label = value
        feature_value_subset = data[data[feature] == value]
        subset_size = feature_value_subset.shape[0]
        feature_value_purity = purity_measure(feature_value_subset)
        weighted_average += (subset_size / total_size) * feature_value_purity
    return subset_purity - weighted_average

def feature_for_split(data, purity_measure):
    # Remove label column
    features = data.drop('label', axis=1)
    # Set to -1 for the case that gain = 0 for some feature
    highest_gain = -1
    purest_feature = None
    for feature in features:
        feature_gain = gain(feature, data, purity_measure)
        if highest_gain < feature_gain:
            highest_gain = feature_gain
            purest_feature = feature
    return purest_feature

def ID3_entropy(data, features, depth):
    # All examples have the same label
    if len(data['label'].unique()) == 1:
        # Return a leaf node with that label
        return Node(label=data['label'].iloc[0])
    
    # Features are empty or depth is reached
    if len(features) == 0 or depth == 0:
        # Return a leaf node with the most common label
        label_count = data['label'].value_counts()
        return Node(label=label_count.idxmax())

    root = Node(None, None, None)
    purest_feature = feature_for_split(data, entropy)
    root.feature = purest_feature
    root.values = {}

    for value in purest_feature:
        # Add a new tree branch for every value
        root.values[value] = None

        # Create a subset Sv of examples in S where A = v
        subset = data[data[purest_feature]] == value

        # S_v is empty
        if subset.shape[0] == 0:
            # Add a leaf node with the most common label in S
            most_common_label = data['label'].value_counts().idxmax()
            root.values[value] = Node(label=most_common_label)
        else:
            # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
            features.remove(purest_feature)
            subtree_node = ID3_entropy(subset, features, None, depth - 1)
            root.values[value] = subtree_node
    return root 

def ID3_majority_error(data, features, label, depth):
    # All examples have the same label
    if len(data['label'].unique()) == 1:
        # Return a leaf node with that label
        return Node(label=data['label'].iloc[0])
    
    # Features are empty or depth is reached
    if len(features) == 0 or depth == 0:
        # Return a leaf node with the most common label
        label_count = data['label'].value_counts()
        return Node(label=label_count.idxmax())

    root = Node(None, None, None)
    purest_feature = feature_for_split(data, majority_error)
    root.feature = purest_feature
    root.values = {}

    for value in purest_feature:
        # Add a new tree branch for every value
        root.values[value] = None

        # Create a subset Sv of examples in S where A = v
        subset = data[data[purest_feature]] == value

        # S_v is empty
        if subset.shape[0] == 0:
            # Add a leaf node with the most common label in S
            most_common_label = data['label'].value_counts().idxmax()
            root.values[value] = Node(label=most_common_label)
        else:
            # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
            features.remove(purest_feature)
            subtree_node = ID3_entropy(subset, features, None, depth - 1)
            root.values[value] = subtree_node
    return root

def ID3_gini_index(data, features, label, depth):
    # All examples have the same label
    if len(data['label'].unique()) == 1:
        # Return a leaf node with that label
        return Node(label=data['label'].iloc[0])
    
    # Features are empty or depth is reached
    if len(features) == 0 or depth == 0:
        # Return a leaf node with the most common label
        label_count = data['label'].value_counts()
        return Node(label=label_count.idxmax())

    root = Node(None, None, None)
    purest_feature = feature_for_split(data, gini_index)
    root.feature = purest_feature
    root.values = {}

    for value in purest_feature:
        # Add a new tree branch for every value
        root.values[value] = None

        # Create a subset Sv of examples in S where A = v
        subset = data[data[purest_feature]] == value

        # S_v is empty
        if subset.shape[0] == 0:
            # Add a leaf node with the most common label in S
            most_common_label = data['label'].value_counts().idxmax()
            root.values[value] = Node(label=most_common_label)
        else:
            # Add the subtree ID3(S_v, features - {purest_feature}) below this branch
            features.remove(purest_feature)
            subtree_node = ID3_entropy(subset, features, None, depth - 1)
            root.values[value] = subtree_node
    return root

class Node:
    def __init__(self, feature, values, label):
        self.feature = feature
        self.values = values # values = {'value': Node}
        self.label = label

def main():
    dataset = pd.read_csv(csv_file_path, header=None)
    dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    features = dataset.drop('label', axis=1)
    tree = ID3_entropy(dataset, features, 6)

if __name__ == "__main__":
    main()