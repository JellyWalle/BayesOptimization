from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from scipy.stats import norm

class AcquisitionFunction(ABC):
    """Базовый класс для функций приобретения."""
    @abstractmethod
    def compute(self, mean: np.ndarray, std: np.ndarray, y_max: float) -> np.ndarray:
        """Вычисляет значение функции приобретения."""
        pass

class UpperConfidenceBound(AcquisitionFunction):
    r"""Upper Confidence Bound: 
    .. math::
        \text{UCB}(x) = \mu(x) + \kappa \sigma(x).
    """
    def __init__(self, kappa: float = 2.576):
        self.kappa = kappa

    def compute(self, mean: np.ndarray, std: np.ndarray, y_max: float) -> np.ndarray:
        return mean + self.kappa * std

class ExpectedImprovement(AcquisitionFunction):
    r"""
    Expected Improvement: 
        .. math::
            \text{EI}(x) = (\mu(x)-y_{\text{max}} - \xi) \Phi\left(
                \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)
                    + \sigma(x) \phi\left(
                    \frac{\mu(x)-y_{\text{max}} -  \xi }{\sigma(x)} \right)

        where :math:`\Phi` is the CDF and :math:`\phi` the PDF of the normal
        distribution.
    """
    def __init__(self, xi: float = 0.01):
        """
        Parameters
        ----------
        xi : float, optional
            Параметр исследования (exploration). 
            Чем больше xi, тем больше алгоритм склонен исследовать области с высокой неопределенностью.
        """
        self.xi = xi

    def compute(self, mean: np.ndarray, std: np.ndarray, y_max: float) -> np.ndarray:
        # Инициализируем массив улучшений нулями
        improvement = np.zeros_like(mean)
        
        # Избегаем деления на ноль в местах, где std == 0
        with np.errstate(divide='ignore'):
            mask = std > 0
            
            if np.any(mask):
                m = mean[mask]
                s = std[mask]
                
                # Z = (mu - y_max - xi) / sigma
                z = (m - y_max - self.xi) / s
                
                # Формула EI
                improvement[mask] = (m - y_max - self.xi) * norm.cdf(z) + s * norm.pdf(z)
                
        return improvement