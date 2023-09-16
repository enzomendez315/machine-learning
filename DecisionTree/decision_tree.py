with open(CSVfile, 'r') as f:
    for line in f:
        terms = line.strip().split(',')