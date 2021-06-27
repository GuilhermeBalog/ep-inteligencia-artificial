knn = {
    "n_neighbors": [5, 4, 6, 7, 3], # pelo menos 5 valores
    "metric": ["euclidean", "manhattan"],
}

mlp = {
    "hidden_layer_sizes": [(6,), (7,), (8,), (9,)]
}

decisionTree = {
    "criterion": ["gini", "entropy"],
    "max_depth": [11, 10, 9, 8, 12],  # pelo menos 5 valores
    "min_samples_leaf": [1, 3, 5, 7, 9] # pelo menos 5 valores
}