import numpy as np

class Bootstrap:

    def __init__(self, X) -> None:
        self.X = np.array(X)

    def calculate_bootstrap(self, bootstraps, estimator):
        self.resampled_stat = []
        for i in range(bootstraps):
            index = np.random.randint(low=0, high=len(self.X), size=len(self.X))
            sample = self.X[index]
            bstatistic = estimator(sample)
            self.resampled_stat.append(bstatistic)
        return np.array(self.resampled_stat)

    def mean(self):
        return np.mean(self.resampled_stat)

    def std(self):
        return np.std(self.resampled_stat)
    
   