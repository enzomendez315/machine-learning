#import csv
import os
import numpy as np
import pandas as pd

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_directory, 'car-4', 'test.csv')

# with open(csv_file_path, 'r') as file:
#     car_reader = csv.reader(file)

#     for line in car_reader:
#         #print(line)
#         #terms = line.strip().split(',')
#         pass

def entropy(data):
    total_size = data.shape[0]
    entropy = 0
    labels = data['label'].unique()
    print('The labels are ', labels, ' and the size is ', len(labels)) # ----------- DELETE LATER
    for value in labels:
        # Create a subset of all rows with label = value
        subset_size = data[data['label'] == value].shape[0]
        if subset_size == 0:
            continue
        print('For the label -> ', value, ' <- the ratio is ', subset_size, '/', total_size) # ----------- DELETE LATER
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

def ID3_entropy(data, attributes, label, depth):
    pass

def ID3_majority_error(data, attributes, label, depth):
    pass

def ID3_gini_index(data, attributes, label, depth):
    pass

def main():
    print("This is the main function.")
    dataset = pd.read_csv(csv_file_path, header=None)
    dataset.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    #total_entropy = entropy(dataset)
    total_ME = majority_error(dataset)
    total_GI = gini_index(dataset)
    total_gain_entropy = gain('doors', dataset, entropy)
    #print('The total entropy is ', total_entropy)
    print('The total majority error is ', total_ME)
    print('The total gini index is ', total_GI)
    print('The total gain is ', total_gain_entropy)

if __name__ == "__main__":
    main()