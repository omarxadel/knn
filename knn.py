from sklearn import datasets
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
                euclidean_distance = math.sqrt((feature[0] - point[0]) ** 2 + (feature[1] - point[1]) ** 2)
                distance.append((euclidean_distance, self.labels[i]))
                i = i + 1
            distance = sorted(distance)[:self.k]
            freq = dict()
            for d in distance:
                if d[1] not in freq:
                    freq[d[1]] = 0
                else:
                    freq[d[1]] = freq[d[1]] + 1
            prediction.append(max(freq, key=freq.get))
        return prediction


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
