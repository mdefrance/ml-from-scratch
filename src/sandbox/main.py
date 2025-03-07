import numpy as np

from model import Kmeans, KNN, NaiveBayes

if __name__ == "__main__":
    print("hello")
    np.random.seed(13)
    x_train = np.random.random_sample((100, 20))
    y_train = np.random.random_integers(0, 1, 100)
    x_dev = np.random.random_sample((1000, 20))
    y_dev = np.random.random_integers(0, 1, 1000)

    model = NaiveBayes()
    model.fit(x_train, y_train)
    # pred = model.predict(x_dev)

    # print(np.mean(np.round(pred) == y_dev))
