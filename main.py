import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parameters import knn, mlp, decisionTree
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

classifiers = [KNeighborsClassifier(), MLPClassifier(), DecisionTreeClassifier()]
params = [knn, mlp, decisionTree]

search_count = 0

def save(title):
    plt.savefig(f'./img/{title}.png')
    plt.clf()


def correlation(df):
    print('Plotando matrix de correlação')

    matrix = df.corr()

    plt.figure()
    sns.heatmap(matrix, cmap='Blues')
    save('correlation')


def pairplot(df):
    print('Plotando pairplot')

    plt.figure()
    sns.pairplot(df, hue='label', corner=True)
    save('pairplot')


def confusion_matrix(name, estimator, X, y):
    print('Plotando matriz de confusão')

    plt.figure()
    plot_confusion_matrix(estimator, X, y, cmap='Blues')
    save(f'{name}')


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


def param_calibration(x, y, iteration):
    # Separa o conjunto de dados em 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    # Define a estratégia de validação cruzada com 10 divisões
    cv = StratifiedKFold(n_splits=10)

    # Itera sobre os classificadores (knn, mlp e decisionTree) e seus parâmetros
    for classifier, param in zip(classifiers, params):
        print("\nClassficador e parametros: ")
        print(classifier, param)
        print()

        # Define a forma de procurar o melhor conjunto de parâmetros
        grid_search = GridSearchCV(
            estimator=classifier,
            param_grid=param,
            # Define as métricas de melhor combinação de classificado/parâmetro
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1,
            cv=cv,
            verbose=0,
            refit='accuracy'
        )
        # Treina o classificador atual com os diferentes conjuntos de parâmetros
        results = grid_search.fit(X_train, y_train)

        # Define a acurácia, precisão, revocação, F-measure e o melhor conjunto de parâmetros
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

        # Cria matriz de confusão
        confusion_matrix(f'{classifier}-{iteration}', results.best_estimator_, X_test, y_test)



def show_dataset_info(x):

    x_info = x.describe()

    # Obtém valores máximos de cada atributo
    x_max_values = np.sort(x_info.loc[['max']].values[0])
    print("Valores máximos por características:")
    print(x_max_values.tolist()[:28])
    print(x_max_values.tolist()[28:])
    print("\nMédia dos valores máximos: ", x_max_values.mean())

    # Obtém média dos valores de cada atributo
    x_mean_values = np.sort(x_info.loc[['mean']].values[0])
    print("\nMédia dos valores por características:")
    print("%.2f " * len(x_mean_values[:28]) % tuple(x_mean_values[:28]))
    print("%.2f " * len(x_mean_values[28:]) % tuple(x_mean_values[28:]))
    print("\nMédia da médias: ", x_mean_values.mean())


if __name__ == '__main__':
    # Configurações de print
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.expand_frame_repr', False)

    # Carrega o conjunto de dados
    x = pd.read_csv('spambase.csv')

    # Cria o pairplot com os atributos mais interessantes
    best_attributes = ['word_freq_000', 'word_freq_your', 'word_freq_you', 'word_freq_email', 'charfreq$', 'label']
    pairplot(x[best_attributes])

    # Cria a matrix de correlação para o conjunto de dados
    correlation(x)

    # Remove a coluna da label para não influenciar no treinamento
    y = x['label']
    x = x.drop('label', axis=1)

    print("----Informações gerais dos dados antes do processamento: -----")
    show_dataset_info(x)
    param_calibration(x.values, y, 1)

    # Normaliza os dados
    normalized_x = normalize(x)
    # Não foi nescessário fazer balanceamento nem lidar com dados faltantes

    print("\n----Informações gerais dos dados depois do processamento: ----")
    show_dataset_info(normalized_x)
    param_calibration(normalized_x.values, y, 2)
