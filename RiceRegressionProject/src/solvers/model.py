
import numpy as np


class LinearRegression():   

    def least_squares(self, X, y):
        X_transpose = np.transpose(X)
        C = np.dot(X_transpose, X)
        C_inverse = np.linalg.inv(C)
        X_pinverse = np.dot(C_inverse, X_transpose)
        w = np.dot(X_pinverse, y)
        return w

    def predict(self, X, y) -> np.array:
        w = self.least_squares(X,y)
        return np.dot(X, w)

class PreProcessing():
    
    def preprocessing(self, X) -> np.array:
        ones = np.ones(len(X))
        X = np.column_stack((ones, X))
        return X