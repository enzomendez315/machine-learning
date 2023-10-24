#!/bin/bash

echo "Running adaboost.py"
python3 adaboost.py

echo "Running bagged_trees.py"
python3 bagged_trees.py

echo "Running random_forest.py"
python3 random_forest.py