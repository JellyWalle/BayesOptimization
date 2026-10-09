"""
реализация BO
"""
import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern

from target_space import TargetSpace
from acquisition_func import UpperConfidenceBound, ExpectedImprovement

class BayesianOptimization:
    """
    Байесовская оптимизация для поиска максимума функции.
    
    Parameters
    ----------
    f : callable
        Функция для максимизации. Принимает гиперпараметры как kwargs, возвращает float.
    pbounds : dict
        Словарь с границами гиперпараметров.
        Пример: {'lr': (1e-5, 1e-2), 'optimizer': ['sgd', 'adam', 'adamw']}
    random_state : int, optional
        Seed для воспроизводимости.
    verbose : int, optional
        Уровень детализации вывода (0 - молча, 1 - кратко, 2 - подробно)
    """
    
    def __init__(self, f, pbounds, random_state=None, verbose=2):
        self.f = f
        self.verbose = verbose
        self.space = TargetSpace(pbounds, random_state)
        
        # GP с ядром Matern
        kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        
        # Функция приобретения
        #self.acquisition = UpperConfidenceBound(kappa=1.576)
        self.acquisition = ExpectedImprovement(xi=0.01)
    
    def suggest(self):
        """Предлагает следующую точку для проверки."""
        if len(self.space) == 0:
            return self.space.random_sample()
        
        # Обучаем GP на уже проверенных точках
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(self.space.params, self.space.target)

        # текущий максимум (нужен для acquisition function: Expected Improvement))
        y_max = self.space.max['target'] if self.space.max is not None else 0.0

        def acq(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            utility = self.acquisition.compute(mu, sigma, y_max)
            return -float(utility[0])
        
        # сначала случайным образом ищем n_random точек
        n_random = 10000
        n_smart = 10

        # вычисляем acquisition function для случайных точек
        candidates = [self.space.random_sample() for _ in range(n_random)]
        candidates_array = np.array([self.space.params_to_array(c) for c in candidates])
        
        acq_values = np.array([acq(c) for c in candidates_array])

        # находим точку с минимальным значением acquisition function
        x_min_r = candidates_array[np.argmin(acq_values)]
        min_acq_r = np.min(acq_values)
        
        # Берем n_smart точек с наименьшими значениями acquisition function для дальнейшей оптимизации
        idxs = np.argsort(acq_values)[:n_smart]
        x_seeds = candidates_array[idxs]
        
        # Определяем, какие параметры непрерывные, а какие дискретные
        continuous_dims = self.space.continuous_dimensions
        continuous_bounds = self.space.bounds[continuous_dims]
        
        min_acq = min_acq_r
        x_min = x_min_r
        
        if all(continuous_dims):
            # Если все параметры непрерывные, то используем L-BFGS-B для уточнения
            for x_try in x_seeds:
                result = minimize(acq, x_try, bounds=self.space.bounds, method='L-BFGS-B')
                if result.success and result.fun < min_acq:
                    x_min = result.x
                    min_acq = result.fun
        else:
            # Если есть дискретные параметры, используем дифференциальную эволюцию для глобальной оптимизации
            # Начальная популяция: 15 * dim случайных точек + seeds
            n_init = 15 * len(self.space.bounds)
            xinit = np.array([
                self.space.params_to_array(self.space.random_sample()) 
                for _ in range(n_init)
            ])
            n_seeds = min(len(x_seeds), len(xinit))
            xinit[:n_seeds] = x_seeds[:n_seeds]
            
            # Seed для DE
            de_seed = self.space.random_state.randint(0, 2**31)
            
            res_de = differential_evolution(
                func=acq,
                bounds=self.space.bounds,
                init=xinit,
                polish=False,
                seed=de_seed,
                maxiter=200
            )
            
            if res_de.success:
                x_min = res_de.x
                min_acq = res_de.fun
                
                # Уточнение непрерывных параметров через L-BFGS-B
                if any(continuous_dims):
                    x_try = x_min.copy()
                    
                    def continuous_acq(x):
                        x_try[continuous_dims] = x
                        return acq(x_try)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = minimize(
                            continuous_acq,
                            x_min[continuous_dims],
                            bounds=continuous_bounds,
                            method='L-BFGS-B'
                        )
                    
                    if result.success and result.fun < min_acq:
                        x_try[continuous_dims] = result.x
                        x_min = x_try
                        min_acq = result.fun
        
        # Clip к границам (защита от численных ошибок)
        x_min = np.clip(x_min, self.space.bounds[:, 0], self.space.bounds[:, 1])

        # Проверяем, не является ли предложенная точка дубликатом
        candidate = self.space.array_to_params(x_min)
        candidate_array = self.space.params_to_array(candidate)
        if self.space.is_duplicate_array(candidate_array):
            for _ in range(500):
                candidate = self.space.random_sample()
                candidate_array = self.space.params_to_array(candidate)
                if not self.space.is_duplicate_array(candidate_array):
                    return candidate
            raise RuntimeError("Не удалось найти новую, неиспользованную точку для BO.")

        return candidate
    
    def probe(self, params):
        """Проверяет точку: вычисляет f(params) и регистрирует результат."""
        array = self.space.params_to_array(params)
        if self.space.is_duplicate_array(array):
            idx = self.space._seen_index[self.space._hash_array(array)]
            target = self.space._target[idx]
        else:
            target = self.f(**params)
            self.space.register(params, target)
        
        if self.verbose >= 1:
            print(f"Проверка: {params}")
            print(f"  Результат: {target:.4f}")
            if self.verbose >= 2 and self.space.max:
                print(f"  Текущий максимум: {self.space.max['target']:.4f}")
            print()

        return target
    
    def maximize(self, init_points=5, n_iter=25):
        """Запускает оптимизацию"""
        if self.verbose >= 1:
            print(f"Старт алгоритма: {init_points} случайных точек + {n_iter} итераций \n")
        
        # случайно инициализируем init_points точек в пространстве гиперпараметров
        for _ in range(init_points):
            params = self.space.random_sample()
            self.probe(params) # проверяем на дубликаты и вычисляем f(params), регистрируем результат
        
        # основной цикл
        for i in range(n_iter):
            if self.verbose >= 1:
                print(f"Итерация {i+1}/{n_iter}")
            params = self.suggest()
            self.probe(params)
        
        if self.verbose >= 1:
            print(f"Лучший результат: {self.space.max['target']:.4f}")
            print(f"Лучшие параметры:")
            for key, val in self.space.max['params'].items():
                print(f"  {key}: {val}")
            print("=" * 50)
    
    @property
    def max(self):
        """Лучшая найденная точка."""
        return self.space.max
    
    @property
    def res(self):
        """Вся история проверок."""
        return self.space.res