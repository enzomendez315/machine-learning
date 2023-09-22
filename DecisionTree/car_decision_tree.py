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

data = pd.read_csv(csv_file_path, header=None)
data.columns = ['buying','maint','doors','persons','lug_boot','safety','label']

def entropy(data):
    total_size = data.shape[0]
    entropy = 0
    labels = data['label'].unique()
    for value in labels:
        # Create a subset of all rows with label = value
        subset_size = data[data['label'] == value].shape[0]
        if subset_size == 0:
            continue
        entropy += - (subset_size / total_size) * np.log2(subset_size / total_size)
    return entropy

def majority_error(data):
    pass

def gini_index(data):
    pass

def gain(data):
    pass

def ID3(data, attributes, label, depth):
    pass




def main():
    print("This is the main function.")

if __name__ == "__main__":
    main()

# total_entropy = entropy(data)
# print('The total entropy is ', total_entropy)