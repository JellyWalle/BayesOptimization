"""
Тест реализации на одномерном примере
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import warnings
warnings.filterwarnings('ignore')
from optimize import BayesianOptimization, UpperConfidenceBound

def target_func(x):
    x = np.asarray(x)
    
    result = np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1 / (x**2 + 1)
    return result
x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target_func(x)
# график целевой функции
plt.figure(figsize=(12, 6))
plt.plot(x, y, linewidth=3, label='target function')
plt.title(
    r'$f(x) = e^{-(x-2)^2} + e^{-(x-6)^2/10} + \frac{1}{x^2+1}$',
    fontsize=16
)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('00_target_func.png', dpi=150)
plt.close()
print(" Сохранён: 00_target_func.png")

def posterior(optimizer, grid):
    """получает у GP предсказание (mu, sigma)"""
    # преобразуем grid в массив и нормализуем его
    grid = np.asarray(grid).reshape(-1)
    grid_norm = np.array([
        optimizer.space.params_to_array({'x': value}) for value in grid
    ]).reshape(-1, 1)
    # берем известные наблюдения и обучаем GP
    optimizer.gp.fit(optimizer.space.params, optimizer.space.target)
    # делаем предсказание на grid
    # mu - среднее предсказание значение функции в каждой точке, 
    # sigma - стандартное отклонение (мера неопределенности предсказания GP)
    mu, sigma = optimizer.gp.predict(grid_norm, return_std=True)
    return np.asarray(mu).ravel(), np.asarray(sigma).ravel()

def plot_gp(optimizer, x, y, step_num):
    """
    функция для построения графика:
    - верхняя панель: целевая функция, наблюдения, предсказание GP, 95% доверительный интервал
    - нижняя панель: функция приобретения (UCB) и следующая точка для проверки
    """
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        f'Gaussian Process and Utility Function After {steps} Steps',
        fontsize=30
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    # Наблюдения
    x_obs = np.array([res["params"]["x"] for res in optimizer.res], dtype=float)
    y_obs = np.array([res["target"] for res in optimizer.res], dtype=float)
    x_plot = np.asarray(x).reshape(-1)
    y_plot = np.asarray(y).reshape(-1)
    
    # Предсказание GP
    mu, sigma = posterior(optimizer, x_plot)
    
    # Верхняя панель: целевая функция + GP
    axis.plot(x_plot, y_plot, linewidth=3, label='Target', color='navy')
    axis.plot(x_obs, y_obs, 'D', markersize=8, label='Observations', color='r')
    axis.plot(x_plot, mu, '--', color='k', label='Prediction')
    axis.fill(
        np.concatenate([x_plot, x_plot[::-1]]),
        np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval'
    )
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})
    
    # Нижняя панель: функция приобретения (UCB)
    # вычисляем UCB напрямую
    x_flat = np.asarray(x).reshape(-1)
    x_flat_norm = np.array([
        optimizer.space.params_to_array({'x': value}) for value in x_flat
    ]).reshape(-1, 1)
    mu_util, sigma_util = optimizer.gp.predict(x_flat_norm, return_std=True)
    mu_util = np.asarray(mu_util).ravel()
    sigma_util = np.asarray(sigma_util).ravel()
    utility = optimizer.acquisition.compute(mu_util, sigma_util, float(np.max(y_obs)))
    
    acq.plot(x_flat, utility, label='Utility Function (UCB)', color='purple', linewidth=2)
    acq.plot(
        x_flat[np.argmax(utility)], np.max(utility), '*', markersize=15,
        label='Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1
    )
    acq.set_xlim((-2, 10))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    filename = f'{step_num + 1:02d}_gp_after_{steps}_steps.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f" Сохранён: {filename}")
optimizer = BayesianOptimization(
    f=target_func,
    pbounds={'x': (-2, 10)},
    random_state=47,
    verbose=0
)

optimizer.acquisition = UpperConfidenceBound(kappa=5.0)

print("\nИнициализируем GP с 2 случайными точками")

optimizer.maximize(init_points=2, n_iter=0)

plot_gp(optimizer, x, y, step_num=0)

# Делаем 7 шагов байесовской оптимизации, по одному за раз, чтобы видеть процесс
for i in range(1, 8):
    print(f"\n=== Шаг BO #{i} ===")
    optimizer.maximize(init_points=0, n_iter=1)
    plot_gp(optimizer, x, y, step_num=i)


best_x = optimizer.max['params']['x']
best_y = optimizer.max['target']

print(f"Найденный максимум: f({best_x:.4f}) = {best_y:.4f}")
print(f"Истинный максимум:  f(2.0000) ≈ {target_func(2.0):.4f}")
print(f"Погрешность по x:   {abs(best_x - 2.0):.4f}")
print(f"Погрешность по f:   {abs(best_y - target_func(2.0)):.4f}")