import numpy as np

class EliminateOutliers:

    def __init__(self, X, upper_bound, lower_bound) -> None:
        self.X = np.array(X)
        self.upper_bound = np.percentile(self.X, upper_bound)
        self.lower_bound = np.percentile(self.X, lower_bound) 
        pass

    
    def filtered_data(self):
        filtered = self.X[(self.X >= self.lower_bound) & (self.X <= self.upper_bound)]
        return filtered