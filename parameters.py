import numpy as np
from abc import ABC, abstractmethod
from typing import Any

class BayesParameter(ABC):
    """Базовый класс для гиперпараметров."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sample(self, random_state: np.random.RandomState) -> Any:
        """Генерирует случайное значение параметра."""
        pass
    
    @abstractmethod
    def to_float(self, value: Any) -> np.ndarray:
        """Преобразует значение в числовой формат для GP."""
        pass
    
    @abstractmethod
    def from_float(self, array: np.ndarray) -> Any:
        """Преобразует числовой формат обратно в значение параметра."""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Размерность параметра в числовом представлении."""
        pass
    
    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Является ли параметр непрерывным."""
        pass
    
    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """Границы параметра в нормализованном виде."""
        pass


class FloatParameter(BayesParameter):
    """Непрерывный параметр."""
    
    def __init__(self, name: str, low: float, high: float, log: bool = False):
        super().__init__(name)
        self.low = low
        self.high = high
        self.log = log
    
    def sample(self, random_state: np.random.RandomState) -> float:
        if self.log:
            return np.exp(random_state.uniform(np.log(self.low), np.log(self.high)))
        return random_state.uniform(self.low, self.high)
    
    def to_float(self, value: float) -> np.ndarray:
        if self.high == self.low:
            return np.array([0.0])
        if self.log:
            return np.array([(np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))])
        return np.array([(value - self.low) / (self.high - self.low)])
    
    def from_float(self, array: np.ndarray) -> float:
        val = float(np.clip(np.asarray(array).reshape(-1)[0], 0.0, 1.0))
        if self.log:
            return float(np.clip(np.exp(val * (np.log(self.high) - np.log(self.low)) + np.log(self.low)), self.low, self.high))
        return float(np.clip(val * (self.high - self.low) + self.low, self.low, self.high))
    
    @property
    def dim(self) -> int:
        return 1
    
    @property
    def is_continuous(self) -> bool:
        return True
    
    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.0, 1.0]])


class IntParameter(BayesParameter):
    """Целочисленный параметр."""
    
    def __init__(self, name: str, low: int, high: int):
        super().__init__(name)
        self.low = low
        self.high = high
    
    def sample(self, random_state: np.random.RandomState) -> int:
        return random_state.randint(self.low, self.high + 1)
    
    def to_float(self, value: int) -> np.ndarray:
        if self.high == self.low:
            return np.array([0.0])
        return np.array([(value - self.low) / (self.high - self.low)])
    
    def from_float(self, array: np.ndarray) -> int:
        val = float(np.clip(np.asarray(array).reshape(-1)[0], 0.0, 1.0))
        return int(np.clip(np.round(val * (self.high - self.low) + self.low), self.low, self.high))
    
    @property
    def dim(self) -> int:
        return 1
    
    @property
    def is_continuous(self) -> bool:
        return False
    
    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.0, 1.0]])


class CategoricalParameter(BayesParameter):
    """Категориальный параметр (one-hot encoding)."""
    
    def __init__(self, name: str, choices: list):
        super().__init__(name)
        self.choices = choices
        self.choice_to_idx = {c: i for i, c in enumerate(choices)}
    
    def sample(self, random_state: np.random.RandomState) -> Any:
        return random_state.choice(self.choices)
    
    def to_float(self, value: Any) -> np.ndarray:
        idx = self.choice_to_idx[value]
        one_hot = np.zeros(len(self.choices))
        one_hot[idx] = 1.0
        return one_hot
    
    def from_float(self, array: np.ndarray) -> Any:
        idx = np.argmax(array)
        return self.choices[idx]
    
    @property
    def dim(self) -> int:
        return len(self.choices)
    
    @property
    def is_continuous(self) -> bool:
        return False
    
    @property
    def bounds(self) -> np.ndarray:
        return np.array([[0.0, 1.0]] * len(self.choices))
