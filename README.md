# Machine Learning
This is a machine learning library developed by Enzo Mendez for CS5350/6350 in University of Utah.


## Decision Tree
The way to create a new tree is by instantiating the DecisionTree class and calling ID3, which is a method that contains 3 parameters: dataset, features, and depth. The dataset parameter should be a pandas dataframe containing all of the features and the training labels. The features parameter represents a dictionary containing the features as keys and its respective values (like 'Sunny', 'Overcast', 'Rainy' for the Outlook feature) as their value pair. When the method is first called, 'features' contains all the available features and with each recursive call, it takes only a subset of the previous features set. The depth parameter represents the maximum depth of the tree and it is used to limit the size of the tree. With each recursive call, the depth decreases by one, and the tree will either stop when it reaches that depth or it will grow to the maximum possible level if the depth is not yet reached.

## Ensemble Learning
Adaboost can be run by instantiating the Adaboost class and calling the method adaboost, which contains 4 parameters: train_dataset, test_dataset, features, and number_classifiers. The train_dataset parameter is the dataset that will be used to train the algorithm and the test_dataset is the dataset for which the algorithm will predict the results based on the training. The features parameter is a list of all the parameters from the datasets, and number_classifiers defines how many trees we want to use for the Adaboost algorithm.

Bagged Trees can be run by instantiating the BaggedTrees class and calling the method bagging, which contains 4 parameters: train_dataset, test_dataset, features, number_of_trees. These are the same parameters that are used in the Adaboost algorithm.

Random Forest can be run by instantiating the RandomForest class and calling the method random_forest, which contains 5 parameters: train_dataset, test_dataset, features, number_of_trees, number_of_features. These are the same parameters that are used in Adaboost and Bagged Trees, with the only difference being the addition of the parameter number_of_features. This parameter defines how many features we want to use to limit the set of features from which we can split at each iteration.

## Perceptron
Perceptron can be run by instantiating the Perceptron class and calling one of three methods to train the algorithm: train_standard, train_voted or train_averaged. Regardless of what method is chosen to train the model, they all have 3 parameters: train_dataset, epochs, learning_rate. The train_dataset parameter is the dataset that will be used to train the model. The parameter epochs represents the number of times that we want to go through all the examples, updating the weights every time. The parameter learning_rate is used to assign the learning rate that will be used to correct mistakes at each iteration. Depending on what method is used to train the model, we have three methods that are used to predict labels for a test_dataset: predict_standard, predict_voted or predict_averaged. All of them use 2 parameters - a dataset that has no labels (so that the model can predict them) and a list of weights for every feature in the dataset.

## SMV