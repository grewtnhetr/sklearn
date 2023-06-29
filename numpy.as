from sklearn.cluster import KMeans
import numpy as np

# Генерация случайных данных
X = np.random.rand(100, 2)

# Кластеризация методом K-средних
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Получение меток кластеров
labels = kmeans.labels_
print(labels)
