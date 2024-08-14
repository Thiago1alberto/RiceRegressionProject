import numpy as np
import scipy.stats as st

class ConfidenceInterval:
    def __init__(self, data, alpha=0.05):
        self.data = data
        self.alpha = alpha
        self.mean = np.mean(self.data) # Média
        self.n = len(self.data) # Tamanho da amostra
        self.std = np.std(self.data, ddof=1)  # Desvio padrão da amostra (usando correção de Bessel)

    def calculate_lower_bound(self):
        # x̄ - (Valor Crítico * Erro Padrão)
        t_score = st.t.ppf(1 - self.alpha / 2, df=self.n)
        margin_of_error = t_score * (self.std / np.sqrt(self.n))
        return self.mean - margin_of_error

    def calculate_upper_bound(self):
        z_score = st.norm.ppf(1 - self.alpha / 2) 
        margin_of_error = z_score * (self.std / np.sqrt(self.n)) 
        return self.mean + margin_of_error

    def calculate_intervals(self):
        lower_bounds = []
        upper_bounds = []
        for i in range(self.coefs.shape[1]):
            coef_i = self.coefs[:, i]
            mean = np.mean(coef_i)
            std_err = st.sem(coef_i)
            interval = st.t.interval(1 - self.alpha, len(coef_i) - 1, loc=mean, scale=std_err)
            lower_bounds.append(interval[0])
            upper_bounds.append(interval[1])
        return lower_bounds, upper_bounds
    
    
    
