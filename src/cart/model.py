""" implementing a decision tree"""

import numpy as np


class DecisionTree:
    """implements a basic decision tree"""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 20) -> None:
        """
        Parameters
        ----------
        max_depth : int, optional
            maximum depth of the tree, by default 5
        min_samples_split : int, optional
            minimum number of observation per split, by default 2
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _compute_gini_impurity(self, y: np.ndarray) -> float:
        """computes gini within a sample"""
        # getting unique values and counts
        _, counts = np.unique(y, return_counts=True)
        freqs = counts / np.sum(counts)
        return 1 - np.sum(freqs) ** 2

    def _compute_total_gini_impurity(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """weighted sum of ginis"""
        # gini per node
        gini_left = self._compute_gini_impurity(y_left)
        gini_right = self._compute_gini_impurity(y_right)

        # weighted sum
        return (len(y_left) * gini_left + len(y_right) * gini_right) / (len(y_left) + len(y_right))

    def _get_best_split(self, x: np.ndarray, y: np.ndarray) -> None:
        """splits data in sub groups"""
        # initiating
        best_gini = 2
        best_split = None

        # iterating over features
        for j in range(x.shape[1]):
            print("getting into feature", j)

            # getting possible threshold for feature j
            x_j = x[:, j]
            thresholds = np.unique(x_j)

            # finding best threshold for feature j
            for threshold in thresholds:

                # splitting data into left and right nodes
                y_left = y[x_j <= threshold]
                y_right = y[x_j > threshold]

                # checking for mininum number of samples
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                # getting gini per samples
                gini = self._compute_total_gini_impurity(y_left, y_right)

                # checking for better split
                if gini < best_gini:
                    best_gini = gini
                    best_split = {"feature": j, "gini": gini, "threshold": threshold}

        return best_split

    def _split_data(
        self, x: np.ndarray, y: np.ndarray, split: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """splits data into left and right sets according to provided split"""

        feature, threshold = split["feature"], split["threshold"]
        left = x[:, feature] <= threshold
        right = x[:, feature] > threshold
        return x[left, :], y[left], x[right, :], y[right]

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int = 0) -> dict:
        """builds the tree"""

        # prediction
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()

        # getting best split for remaining data
        best_split = self._get_best_split(x, y)

        # if no split found returning prediction
        if not best_split:
            return np.bincount(y).argmax()

        # splitting according to found best split
        x_left, y_left, x_right, y_right = self._split_data(x, y, best_split)

        # getting best split for each side
        return {
            **best_split,
            "left": self._build_tree(x_left, y_left, depth=depth + 1),
            "right": self._build_tree(x_right, y_right, depth=depth + 1),
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """fits decision tree"""

        self.tree = self._build_tree(x, y)

    def _predict_sample(self, x: np.ndarray, node: dict) -> int:
        """prediction for one sample"""

        # checking that there are child nodes
        if isinstance(node, dict):
            # checking for left or right prediction
            if x[node["feature"]] <= node["threshold"]:
                return self._predict_sample(x, node["left"])
            return self._predict_sample(x, node["right"])

        # returning final node class
        return node

    def predict(self, x: np.ndarray) -> np.ndarray:
        """predicts on all samples"""

        return np.array([self._predict_sample(sample, self.tree) for sample in x])


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(50)
    X_train = np.random.rand(1000, 5)  # 100 samples, 2 features
    y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(int)  # Binary target (simple rule)
    y_train = np.random.randint(0, 2, 1000)
    # print(y_train)

    X_test = np.random.rand(5, 5)  # 5 test samples

    # Train and predict with the decision tree
    tree = DecisionTree(max_depth=10, min_samples_split=2)
    tree.fit(X_train, y_train)

    predictions = tree.predict(X_test)
    print("Predictions:", predictions)
    print("tree", tree.tree)
