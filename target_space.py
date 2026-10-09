import numpy as np
from parameters import *

class TargetSpace:
    """Хранит пространство поиска и историю проверенных точек."""
    
    def __init__(self, pbounds: dict, random_state=None):
        self.random_state = np.random.RandomState(random_state)
        
        # Создаём объекты параметров на основе pbounds
        self.params_config = {}
        for name, bounds in pbounds.items():
            if isinstance(bounds, BayesParameter):
                self.params_config[name] = bounds
            elif isinstance(bounds, tuple) and len(bounds) == 2:
                low, high = bounds
                if isinstance(low, (int, np.integer)) and isinstance(high, (int, np.integer)):
                    self.params_config[name] = IntParameter(name, int(low), int(high))
                else:
                    self.params_config[name] = FloatParameter(name, float(low), float(high))
            elif isinstance(bounds, tuple) and len(bounds) == 3 and bounds[-1] in (int, np.integer):
                # Поддержка формата (low, high, int)
                low, high, _ = bounds
                self.params_config[name] = IntParameter(name, int(low), int(high))
            elif isinstance(bounds, list):
                # Категориальный параметр
                self.params_config[name] = CategoricalParameter(name, bounds)
            else:
                raise ValueError(f"Неподдерживаемый формат для параметра {name}: {bounds}")
        
        # список параметров
        self.keys = list(self.params_config.keys())
        
        # размерность пространства
        self.dim = sum(p.dim for p in self.params_config.values())
        
        # Создаём маски для каждого параметра
        self.masks = {}
        pos = 0
        for key in self.keys:
            p = self.params_config[key]
            mask = np.zeros(self.dim, dtype=bool)
            mask[pos:pos + p.dim] = True
            self.masks[key] = mask
            pos += p.dim
        
        # Вычисляем границы пространства (в нормализованном виде)
        self._bounds = self._calculate_bounds()
        
        # История проверенных точек
        self._params = np.empty((0, self.dim))
        self._target = np.empty(0)
        self._seen = set()
        self._seen_index = {}

    def __contains__(self, array):
        """Проверка на уже вычисленную точку."""
        key = self._hash_array(np.asarray(array, dtype=float))
        return key in self._seen
    
    def _calculate_bounds(self) -> np.ndarray:
        """Вычисляет границы пространства в нормализованном виде."""
        bounds = np.empty((self.dim, 2))
        for key in self.keys:
            bounds[self.masks[key]] = self.params_config[key].bounds
        return bounds
    
    @property
    def bounds(self) -> np.ndarray:
        """Границы пространства поиска."""
        return self._bounds
    
    @property
    def continuous_dimensions(self) -> np.ndarray:
        """Маска непрерывных измерений."""
        result = np.zeros(self.dim, dtype=bool)
        for key in self.keys:
            result[self.masks[key]] = self.params_config[key].is_continuous
        return result
    
    def __len__(self):
        return len(self._target)
    
    @property
    def params(self):
        return self._params
    
    @property
    def target(self):
        return self._target
    
    @staticmethod
    def _hash_array(array: np.ndarray):
        """Хэширует точку в нормализованном пространстве с фиксированной точностью."""
        arr = np.asarray(array, dtype=float).ravel()
        return tuple(np.round(arr, 12).tolist())

    def is_duplicate_array(self, array: np.ndarray) -> bool:
        """Проверяет, была ли точка уже оценена."""
        key = self._hash_array(np.asarray(array, dtype=float))
        return key in self._seen

    def params_to_array(self, params: dict) -> np.ndarray:
        """Преобразует словарь параметров в числовой массив."""
        result = np.zeros(self.dim)
        for key in self.keys:
            p = self.params_config[key]
            result[self.masks[key]] = p.to_float(params[key])
        return result
    
    def array_to_params(self, array: np.ndarray) -> dict:
        """Преобразует числовой массив в словарь параметров."""
        result = {}
        for key in self.keys:
            p = self.params_config[key]
            result[key] = p.from_float(array[self.masks[key]])
        return result
    
    def random_sample(self) -> dict:
        """Генерирует случайную точку в пространстве поиска."""
        result = {}
        for key in self.keys:
            result[key] = self.params_config[key].sample(self.random_state)
        return result
    
    def register(self, params: dict, target: float):
        """Регистрирует проверенную точку. Дубликаты не добавляются."""
        array = self.params_to_array(params)
        key = self._hash_array(array)
        if key in self._seen:
            idx = self._seen_index[key]
            return self._target[idx]

        self._seen.add(key)
        self._seen_index[key] = len(self._target)
        self._params = np.vstack([self._params, array]) if len(self._params) > 0 else array.reshape(1, -1)
        self._target = np.append(self._target, target)
        return target
    
    @property
    def max(self):
        """Возвращает лучшую найденную точку."""
        if len(self) == 0:
            return None
        idx = np.argmax(self._target)
        return {
            'params': self.array_to_params(self._params[idx]),
            'target': self._target[idx]
        }
    
    @property
    def res(self):
        """Возвращает всю историю проверок."""
        return [
            {'params': self.array_to_params(p), 'target': t}
            for p, t in zip(self._params, self._target)
        ]