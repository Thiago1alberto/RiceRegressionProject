import numpy as np

class Bootstrap:

    def __init__(self, X, y) -> None:
        self.X = np.array(X)
        self.y = np.array(y)
        self.results = []

    def calculate_bootstrap(self, boots, estimator):
        n_samples = len(self.X)
        self.resampled_stat = []
        for i in range(boots):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sample_X = self.X[indices]
            sample_y = self.y[indices]
            wi = estimator.least_squares(sample_X, sample_y)
            self.results.append(wi)
        return np.array(self.results)