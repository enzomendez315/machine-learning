import csv
import os

# Get the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
csv_file_path = os.path.join(script_directory, 'car-4', 'test.csv')

with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        print(line)
        # terms = line.strip().split(',')

# print(terms[3])