import os
import numpy as np
import pandas as pd

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

    def ID3_majority_error(self, data, features, depth):
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
        purest_feature = self.feature_for_split(data, self.majority_error)
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
                subtree_node = self.ID3_majority_error(subset, features, depth - 1)
                root.values[value] = subtree_node
        return root

    def ID3_gini_index(self, data, features, depth):
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
        purest_feature = self.feature_for_split(data, self.gini_index)
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
                subtree_node = self.ID3_gini_index(subset, features, depth - 1)
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
    car_file_path = os.path.join(script_directory, '..', 'Datasets', 'car-4', 'train.csv')
    tennis_file_path = os.path.join(script_directory, '..', 'Datasets', 'tennis', 'train.csv')
    bank_file_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')

    # Using car dataset
    car_dataset = pd.read_csv(car_file_path, header=None)
    car_dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    features = car_dataset.drop('label', axis=1)
    DT = DecisionTree()
    car_tree = DT.ID3_entropy(car_dataset, features, 3)
    DT.print_tree(car_tree)

    # Using tennis dataset
    tennis_dataset = pd.read_csv(tennis_file_path, header=None)
    tennis_dataset.columns = ['Outlook','Temp','Humidity','Wind','label']
    debug_features = tennis_dataset.drop('label', axis=1)
    tennis_tree = DT.ID3_entropy(tennis_dataset, debug_features, 3)
    DT.print_tree(tennis_tree)

    # Using bank dataset
    bank_dataset = pd.read_csv(bank_file_path, header=None)
    bank_dataset.columns = ['age','job','marital','education',
                       'default','balance','housing', 'loan', 
                       'contact', 'day', 'month', 'duration', 
                       'campaign', 'pdays', 'previous', 'poutcome', 'label']
    features = bank_dataset.drop('label', axis=1)
    numerical_features = {'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'}
    bank_tree = DT.ID3_entropy_numerical(bank_dataset, features, numerical_features, 6)
    DT.print_tree(bank_tree)

if __name__ == "__main__":
    main()