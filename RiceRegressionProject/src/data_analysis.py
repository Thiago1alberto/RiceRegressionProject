import numpy  as np
import scipy.stats as st

class DataAnalysis:
    

    def __init__(self, X) -> None:
        self.X = np.array(X)
    
    def standard_deviation(self):
        return np.std(self.X)
    
    def mean(self):
        return np.mean(self.X)
    
    def cdf_value(self,x):
        return st.norm.cdf(x, loc=self.mean(), scale=self.standard_deviation())