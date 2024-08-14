import numpy as np

class LinearRegression:
    
    """
    Classe para realizar a regressão linear usando a equação normal.

    Métodos:
    --------
    fit(X, y):
        Ajusta o modelo aos dados fornecidos.
    
    predict(X):
        Faz previsões usando o modelo ajustado.
    """
    def __init__(self):
        self.coef_ = None #Matriz de entrada (Coeficientes do modelo)
        self.intercept_ = None #Vetor de saída (Intercepto do modelo)

    def fit(self, X, y):
        """
        Ajusta o modelo de regressão linear aos dados de treinamento.

        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Matriz de características de entrada (variáveis independentes).
        
        y : array-like, shape (n_samples,)
            Vetor de saída (variável dependente).
        
        Retorna:
        --------
        self : returns an instance of self.
        """
        # Adiciona uma coluna de 1s para o intercepto
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Calcula os coeficientes utilizando a equação normal
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 
        # X_b.T é a transposta de X_b e X_b.T.dot(X_b) realiza a multiplicação matricial.
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        return self

    def predict(self, X):
        """
        Faz previsões usando o modelo ajustado.

        Parâmetros:
        -----------
        X : Matriz de características de entrada (variáveis independentes).
        
        Retorna:
        --------
        y_pred : Vetor de previsões.
        
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])