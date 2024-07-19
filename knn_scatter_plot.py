import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from collections import Counter

# Carregar o Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Dividir os dados em conjuntos de treino e teste (70% treino, 30% teste)
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Função para calcular a distância Euclidiana entre dois pontos
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Função k-NN
def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        # Calcular a distância de test_point para todos os pontos de treino
        distances = []
        for i, train_point in enumerate(X_train):
            distance = euclidean_distance(test_point, train_point)
            distances.append((distance, y_train[i]))
        
        # Ordenar as distâncias e pegar as k mais próximas
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        
        # Pegar as classes dos k vizinhos mais próximos
        k_nearest_labels = [label for _, label in k_nearest_neighbors]
        
        # Determinar a classe mais comum (votação)
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])
    
    return np.array(y_pred)

# Prever os rótulos para o conjunto de teste
y_pred = knn_predict(X_train, y_train, X_test, k=3)

# Calcular a precisão do modelo
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Acurácia do modelo k-NN: {accuracy * 100:.2f}%')

# Gráfico 1: Dispersão das diferentes espécies
plt.figure(figsize=(12, 6))

# Usando apenas as duas primeiras características para visualização
plt.subplot(1, 2, 1)
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Dispersão das Espécies')
plt.legend()

# Gráfico 2: Inferências corretas e erradas
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=np.where(y_pred == y_test, 'g', 'r'), marker='o', label='Predição')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.title('Inferências Corretas (Verde) e Erradas (Vermelho) para k=3')
plt.legend()

plt.tight_layout()
plt.show()
