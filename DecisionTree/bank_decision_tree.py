import csv
import os

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_directory, 'bank-4', 'test.csv')

with open(csv_file_path, 'r') as file:
    car_reader = csv.reader(file)

    for line in car_reader:
        #print(line)
        #terms = line.strip().split(',')
        pass


def entropy():
    pass

def majority_error():
    pass

def gini_index():
    pass

def ID3(data, attributes, label, depth):
    pass