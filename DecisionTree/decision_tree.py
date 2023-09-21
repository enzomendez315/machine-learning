import csv

with open('test.csv', 'r') as file:
    csv_reader = csv.reader(file)

    for line in csv_reader:
        terms = line.strip().split(',')



print(terms[3])