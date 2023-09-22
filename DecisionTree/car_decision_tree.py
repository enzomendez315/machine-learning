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

def gain(data):
    pass

def ID3(data, attributes, label, depth):
    pass




def main():
    print("This is the main function.")
    data = pd.read_csv(csv_file_path, header=None)
    data.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    total_entropy = entropy(data)
    print('The total entropy is ', total_entropy)

if __name__ == "__main__":
    main()