import numpy as np
import scipy.stats as st

class BootstrapRegression:
    """
    Classe para realizar regressão linear com intervalo de confiança via bootstrap.

    Atributos:
    ----------
    matrix : Matriz de características de entrada (variáveis independentes).
    
    target : Vetor de saída (variável dependente).
    
    bootstrap_coefs : Lista de coeficientes gerados em cada amostra bootstrap.

    Métodos:
    --------
    calculate_bootstrap_by_double_index(bootstraps, estimator):
        Realiza o bootstrap e ajusta o modelo nas amostras geradas.
    
    calculate_confidence_interval(alpha=0.05):
        Calcula o intervalo de confiança para os coeficientes utilizando as amostras bootstrap.
    """
    def __init__(self, matrix, target):
        self.matrix = matrix
        self.target = target
        self.bootstrap_coefs = []

    def calculate_bootstrap_by_double_index(self, bootstraps, estimator):
        """
        Realiza o bootstrap e ajusta o modelo nas amostras geradas.

        Parâmetros:
        -----------
        bootstraps : Número de amostras bootstrap a serem geradas.
        
        estimator : Classe do estimador a ser ajustado (por exemplo, LinearRegression).

        Retorna:
        --------
        bootstrap_coefs : Matriz contendo os coeficientes bootstrap para cada amostra.
        """
        for _ in range(bootstraps):
            # Cria um array de índices para a matriz original
            index = np.arange(len(self.matrix))
            # Gera índices resampleados com reposição
            resample_index = np.random.choice(index, size=len(index), replace=True)
            # Cria as amostras bootstrap para X e y usando os índices resampleados
            X_resampled = self.matrix[resample_index]
            y_resampled = self.target[resample_index]
            # Ajusta o modelo estimador às amostras bootstrap
            model = estimator().fit(X_resampled, y_resampled)
            # Armazena os coeficientes (intercepto e coeficientes) do modelo ajustado
            self.bootstrap_coefs.append(np.r_[model.intercept_, model.coef_])
        return np.array(self.bootstrap_coefs)

    def calculate_confidence_interval(self, alpha=0.05):
        """
        Calcula o intervalo de confiança para os coeficientes utilizando as amostras bootstrap.

        Parâmetros:
        -----------
        alpha : Nível de significância para o intervalo de confiança (ex: 0.05 para IC de 95%).

        Retorna:
        --------
        lower_bounds : Limite inferior do intervalo de confiança para cada coeficiente.
        
        upper_bounds : Limite superior do intervalo de confiança para cada coeficiente.
        
        means : Valor médio dos coeficientes bootstrap para cada coeficiente.
        """
         # Converte a lista de coeficientes bootstrap para um array numpy
        coefs = np.array(self.bootstrap_coefs)
        # Calcula os limites inferiores do intervalo de confiança
        lower_bounds = np.percentile(coefs, alpha/2*100, axis=0)
        # Calcula os limites superiores do intervalo de confiança
        upper_bounds = np.percentile(coefs, (1 - alpha/2)*100, axis=0)
        # Calcula a média dos coeficientes bootstrap
        means = np.mean(coefs, axis=0)
        return lower_bounds, upper_bounds, means