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
    
    def predict(self, tree, data):
        for index, _ in data.iterrows():
            self._predict_label(tree, data, index)
        return data
    
    def _predict_label(self, tree, data, row_index):
        # Leaf node indicates there is a label
        if not tree.values:
            data.at[row_index, 'label'] = tree.label
            return
        
        # Get the feature value using the feature found in tree
        feature_value = data.at[row_index, tree.feature]
        subtree = tree.values[feature_value]
        self._predict_label(subtree, data, row_index)

    def prediction_error(self, actual_labels, predicted_labels):
        counter = 0
        for i in range(len(actual_labels)):
            if actual_labels[i] != predicted_labels[i]:
                counter = counter + 1
        return counter / len(actual_labels)

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
    def __init__(self, feature, values, label):
        self.feature = feature
        self.values = values    # values = {'value': Node}
        self.label = label

def main():
    # Get the directory of the script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the full path to the CSV files
    car_train_path = os.path.join(script_directory, '..', 'Datasets', 'car-4', 'train.csv')
    tennis_train_path = os.path.join(script_directory, '..', 'Datasets', 'tennis', 'train.csv')
    bank_train_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'train.csv')
    car_test_path = os.path.join(script_directory, '..', 'Datasets', 'car-4', 'test.csv')
    bank_test_path = os.path.join(script_directory, '..', 'Datasets', 'bank-4', 'test.csv')

    DT = DecisionTree()

    # Using car dataset
        # Upload training dataset
    car_train_dataset = pd.read_csv(car_train_path, header=None)
    car_train_dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    car_features = car_train_dataset.drop('label', axis=1)
        # Upload testing dataset
    car_test_dataset = pd.read_csv(car_test_path, header=None)
    car_test_dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
        # Create copy of testing dataset for predicting
    car_predicted_dataset = pd.DataFrame(car_test_dataset)
    car_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
        # Construct the tree, predict and compare
    car_tree = DT.ID3_entropy(car_train_dataset, car_features, 3)
    car_predicted_dataset = DT.predict(car_tree, car_predicted_dataset)
    car_error = DT.prediction_error(car_test_dataset['label'].to_numpy(), car_predicted_dataset['label'].to_numpy())
    DT.print_tree(car_tree)
    print('The prediction error for this tree is', car_error)

    # Using tennis dataset
        # Upload training dataset
    tennis_train_dataset = pd.read_csv(tennis_train_path, header=None)
    tennis_train_dataset.columns = ['Outlook','Temp','Humidity','Wind','label']
    tennis_features = tennis_train_dataset.drop('label', axis=1)
        # Upload testing dataset
    tennis_test_dataset = pd.read_csv(tennis_train_path, header=None)
    tennis_test_dataset.columns = ['Outlook','Temp','Humidity','Wind','label']
        # Create copy of testing dataset for predicting
    tennis_predicted_dataset = pd.DataFrame(tennis_test_dataset)
    tennis_predicted_dataset['label'] = ""   # or = np.nan for numerical columns
        # Construct the tree, predict and compare
    tennis_tree = DT.ID3_entropy(tennis_train_dataset, tennis_features, 3)
    tennis_predicted_dataset = DT.predict(tennis_tree, tennis_predicted_dataset)
    tennis_error = DT.prediction_error(tennis_test_dataset['label'].to_numpy(), tennis_predicted_dataset['label'].to_numpy())
    DT.print_tree(tennis_tree)
    print('The prediction error for this tree is', tennis_error)

    # Using bank dataset
    bank_dataset = pd.read_csv(bank_train_path, header=None)
    bank_dataset.columns = ['age','job','marital','education',
                       'default','balance','housing', 'loan', 
                       'contact', 'day', 'month', 'duration', 
                       'campaign', 'pdays', 'previous', 'poutcome', 'label']
    bank_features = bank_dataset.drop('label', axis=1)
    numerical_features = {'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'}
    bank_tree = DT.ID3_entropy_numerical(bank_dataset, bank_features, numerical_features, 6)
    DT.print_tree(bank_tree)

if __name__ == "__main__":
    main()