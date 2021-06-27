knn = {
    "n_neighbors": [5, 4, 6, 7, 3], # Quantidade de vizinhos
    "metric": ["euclidean", "manhattan"], # Tipo da fórmula de distância utilizada
}

mlp = {
    "hidden_layer_sizes": [(6,), (7,), (8,), (9,)] # Número de neurônios na camada escondida
}

decisionTree = {
    "criterion": ["gini", "entropy"],
    "max_depth": [11, 10, 9, 8, 12],
    "min_samples_leaf": [1, 3, 5, 7, 9] # Número mínimo de amostras para ser uma folha
}