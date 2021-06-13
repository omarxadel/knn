from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math
import pandas as pd
import numpy as np


class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, unlabeled_data):
        distance = []
        prediction = []
        for point in unlabeled_data:
            i = 0
            for feature in self.data:
                euclidean_distance = 0
                for j in range(len(feature)):
                    euclidean_distance += ((feature[j] - point[j]) ** 2)
                distance.append((math.sqrt(euclidean_distance), self.labels[i]))
                i = i + 1
            distance = sorted(distance, key=lambda x: x[0])[:self.k]
            freq = dict()
            for d in distance:
                if d[1] not in freq:
                    freq[d[1]] = 0
                else:
                    freq[d[1]] = freq[d[1]] + 1
            prediction.append(max(freq, key=freq.get))
        return prediction

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(y_pred == y)/len(y)


if __name__ == '__main__':
    # LOAD DATA SET
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    df = pd.DataFrame(X, columns=iris.feature_names)

    # PERFORM KNN

    knn = KNN(3)
    knn.fit(X, y)
    X_new = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])
    prediction = knn.predict(X_new)
    for i in range(len(prediction)):
        print(iris.target_names[prediction[i]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    knn = KNN(7)
    knn.fit(X_train, y_train)
    print(str(knn.score(X_test, y_test) * 100) + '%')

    knn_2 = KNeighborsClassifier(n_neighbors=7)
    knn_2.fit(X_train, y_train)
    print(str(knn_2.score(X_test, y_test) * 100) + '%')
