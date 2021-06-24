import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import knn, mlp, decisionTree
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

classifiers = [KNeighborsClassifier(), MLPClassifier(), DecisionTreeClassifier()]
params = [knn, mlp, decisionTree]


def save(title):
  plt.savefig(f'./img/{title}.png')
  plt.clf()


def correlation(df):
  print('Plotando matrix de correlação')

  matrix = df.corr()
  c = df.corr().abs()

  s = c.unstack()
  so = s.sort_values(kind="quicksort")
  print(so)

  plt.figure()
  sns.heatmap(matrix, cmap='Greens')
  save('correlation')


def pairplot(df):
  print('Plotando pairplot')

  plt.figure()
  sns.pairplot(df, hue='label', corner=True)
  save('pairplot')


def normalize(X):
    df_final = []

    # iterar sobre cada coluna
    for (feature, data) in X.iteritems():
        column = []
        maxVal = data.max()
        minVal = data.min()

        diff = maxVal - minVal
        # iterar sobre cada valor em cada coluna
        for i in data:
            L = (i - minVal)/diff
            column.append(L)
        df_final.append(column)

    X_novo = pd.DataFrame(df_final).T
    X_novo.columns = X.columns
    return X_novo


def param_calibration(x, y):
    cv = StratifiedKFold(n_splits=10)

    for classifier, param in zip(classifiers, params):
        print("\nClassficador e parametros: ")
        print(classifier, param)
        print("")

        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1,
            cv=cv,
            verbose=0,
            refit='accuracy'
        )
        results = grid_search.fit(x, y)

        accuracy = results.best_score_
        precision = results.cv_results_['mean_test_precision'][results.best_index_]
        recall = results.cv_results_['mean_test_recall'][results.best_index_]
        f1 = results.cv_results_['mean_test_f1'][results.best_index_]
        parameters = results.best_params_

        print("Acurácia: ", accuracy)
        print("Precisão: ", precision)
        print("Revocação (sensibilidade): ", recall)
        print("F-measure: ", f1)
        print("Melhores parametros: ", parameters)


def show_dataset_info(x):

    x_info = x.describe()

    print("Informações gerais sobre o conjunto de dados: ")
    print(x_info)

    x_max_values = np.sort(x_info.loc[['max']].values[0])
    print("Valores máximos por características:")
    print(x_max_values.tolist()[:28])
    print(x_max_values.tolist()[28:])
    print("\nMédia dos valores máximos: ", x_max_values.mean())

    x_mean_values = np.sort(x_info.loc[['mean']].values[0])
    print("\nMédia dos valores por características:")
    print("%.2f " * len(x_mean_values[:28]) % tuple(x_mean_values[:28]))
    print("%.2f " * len(x_mean_values[28:]) % tuple(x_mean_values[28:]))
    print("\nMédia da médias: ", x_mean_values.mean())


if __name__ == '__main__':

    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.expand_frame_repr', False)

    x = pd.read_csv('spambase.csv')

    y = x['label']
    x = x.drop('label', axis=1)

    print("----Informações gerais dos dados antes do processamento: -----")
    show_dataset_info(x)
    param_calibration(x.values, y)

    normalized_x = normalize(x)
    print("\n----Informações gerais dos dados depois do processamento: ----")
    show_dataset_info(normalized_x)
    param_calibration(normalized_x.values, y)