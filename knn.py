from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
import matplotlib.pyplot as plt


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
            distance = []
            for feature in self.data:
                euclidean_distance = 0
                for j in range(len(feature)):
                    euclidean_distance += ((feature[j] - point[j]) ** 2)
                euclidean_distance = math.sqrt(euclidean_distance)
                lab = self.labels[i]
                distance.append((euclidean_distance, lab))
                i = i + 1
            distance.sort(key=lambda tup: tup[0])
            distance = distance[:self.k]
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
        return np.sum(y_pred == y) / len(y)


if __name__ == '__main__':
    # Load data set
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Get train data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Setup arrays to store train and test accuracies
    neighbors = np.arange(1, 25)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over different values of k
    for i, k in enumerate(neighbors):
        # Setup a k-NN Classifier with k neighbors: knn
        knn = KNN(k=k)

        # Fit the classifier to the training data
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        # Compute accuracy on the testing set
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    plt.title('k-NN: Varying Number of Neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
