from statistics import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

class Plotter:
    def __init__(self, data=None):
        self.data = data
    
    def plot_hist(self, xlabel, ylabel, bins=100):
        plt.hist(self.data, bins=bins)
        plt.title('Histogram')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    
    def plot_scatter(self, y=None, alpha=.3):
        if y is not None:
            plt.scatter(self.data, y,alpha=alpha)
        else:
            plt.scatter(range(len(self.data)), self.data)
        plt.title('Scatter Plot')
        plt.xlabel('Index')
        plt.ylabel('Data')
        plt.show()

    def regression_plot(self, y, predicted, alpha=1, x_label='X', y_label='y'):
        plt.scatter(self.data, y, color='blue', label='Real Data', alpha=alpha)
        plt.plot(self.data, predicted, color='red', label='Linear Regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Regressão Linear')
        plt.legend()
        plt.show()

    def plot_ci(upper_data, lower_data, xlabel, ylabel, title):
        plt.errorbar(x=np.mean(upper_data), y=0, xerr=[[np.min(lower_data)], [np.max(upper_data)]], fmt='o', color='skyblue', linestyle='-')
        plt.yticks([0], [ylabel], fontsize=12)
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_confidance_interval(self, lower_bound, upper_bound, xlabel, ylabel, title):
        x = np.arange(len(lower_bound))
        plt.errorbar(x, (upper_bound + lower_bound) / 2, yerr=[(upper_bound - lower_bound) / 2], fmt='o', ecolor='g', capsize=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()
        
    


    def linear_plot(X, y, w):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Dados')
        plt.plot(X, LinearRegression().fit(X, y).predict(X), color='red', label='Regressão')
        plt.xlabel('Area')
        plt.ylabel('Major Axis Length')
        plt.title('Regressão Linear entre Area e Major Axis Length')
        plt.legend()
        plt.grid(True)
        plt.show()

   



class ConfidanceInterval:

    import numpy as np
    import matplotlib.pyplot as plt
    
    
    def confidance_interval(w0, w1, ci_w0, ci_w1):
        mean_w0 = np.mean(w0)
        mean_w1 = np.mean(w1)

        error_w0 = [[mean_w0 - ci_w0[0]], [ci_w0[1] - mean_w0]]
        error_w1 = [[mean_w1 - ci_w1[0]], [ci_w1[1] - mean_w1]]

        error_w0 = np.array(error_w0).reshape(2, 1)
        error_w1 = np.array(error_w1).reshape(2, 1)

        plt.figure(figsize=(10, 6))
        plt.errorbar(0, np.mean(w0), yerr=np.array([[np.mean(w0) - ci_w0[0]], [ci_w0[1] - np.mean(w0)]]),
                    fmt='o', capsize=5, label=f'Média: {np.mean(w0):.2f}')
        plt.vlines(0, ci_w0[0], ci_w0[1], color=None, linestyle='-', linewidth=2, label='IC 95%')
        plt.vlines(0, np.min(w0), np.min(w0), label=f'lower_bound {np.round(ci_w0[0],3)}')
        plt.vlines(0, np.max(w0), np.max(w0), label=f'upper_bound {np.round(ci_w0[1],3)}')
        plt.vlines(0, np.min(w0), np.max(w0), color='green', linestyle='--', linewidth=2, label='Min/Max')

        plt.xticks([0], ['w0 (Intercepto)'])
        plt.ylabel('Valor')
        plt.title(f'Intervalo de Confiança para w0')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.errorbar(0, np.mean(w1), yerr=np.array([[np.mean(w1) - ci_w1[0]], [ci_w1[1] - np.mean(w1)]]),
                    fmt='o', capsize=5, label=f'Média: {np.mean(w1):.2f}')
        plt.vlines(0, ci_w1[0], ci_w1[1], label='IC 95%')
        plt.vlines(0, np.min(w1), np.min(w1), label=f'lower_bound {np.round(ci_w1[0],3)}')
        plt.vlines(0, np.max(w1), np.max(w1), label=f'upper_bound {np.round(ci_w1[1],3)}')
        plt.vlines(0, np.min(w1), np.max(w1), color='green', linestyle='--', linewidth=2, label='Min/Max')
        plt.xticks([0], ['w1 (Coeficiente)'])
        plt.ylabel('Valor')
        plt.title(f'Intervalo de Confiança para w1')
        plt.legend()
        plt.grid(True)
        plt.show()