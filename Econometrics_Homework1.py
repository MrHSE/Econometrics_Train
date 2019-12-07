# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:53:19 2019
@author: Daniil
"""

from numpy import dot, array, eye, ones, hstack, vstack, mean, transpose, corrcoef, diagonal, reshape, exp, full, diagflat
from numpy.random import normal, uniform, standard_cauchy
from numpy.linalg import inv, det
import math
from statsmodels.stats.stattools import jarque_bera
import random

# Создадим файл, в который будем писать все результаты
file = open('Econometrics_Homework1_results.txt', 'w')

file.write('Упражнения на использование метода Монте-Карло с различными DGP.\n')
file.write('Задача 1. Генерация моделей, проверка свойств.\n')
file.write('Симуляция проводится для вычисления E(b|X) и распределения t-отношения как распределения, условного относительно X.\n')
file.write('Сгенерируем процесс вида: y = b1 + b2*xi + eps, где xi = r*x0 + d + A * ita;...\n')

# Генерация входных значений
fi = 0.6
C = 2
b1 = 1
b2 = 0.5

# Возможные значения размеров выборок
var_lengths = [32, 50, 100, 200]
# Критические значения t-статистики для соответствующих выборок
t_crits = [2.0395, 2.0096, 1.9842, 1.972]
iterations = int(input('Введите количество итераций для рассчитываемых моделей: '))

# Задание на стр 107, часть 1
file.write('\n\n\nНачало генерации моделей с детерминированным x\n\n')
for length_num, length in enumerate(var_lengths):
    # ---------------Генерация необходимых матриц для расчёта------------------------
    file.write('Сначала для рассматриваемых моделей (с разным количеством наблюдений) X генерируется только один раз.\n')
    A = eye(var_lengths[length_num])
    for k in range(1, var_lengths[length_num]):
        for i in range(k, var_lengths[length_num]):
            A[i][k - 1] = fi ** (i - (k - 1))
    # Компоненты для генерации x
    ita = normal(0, 1, size=(var_lengths[length_num], 1))
    r = array([[fi ** _] for _ in range(var_lengths[length_num])])
    x0 = normal(C / (1 - fi), 1 / (1 - fi ** 2))
    # Генерация d
    d = ones((var_lengths[length_num], 1))
    for _ in range(var_lengths[length_num]):
        for __ in range(_):
            d[_][0] += fi ** (__ + 1)    
    # Генерация x
    x = r * x0 + d + dot(A, ita)
    const = array([[1] for _ in range(var_lengths[length_num])])
    # Матрица значений X
    X = hstack((const, x))
    # Создание массива для сбора рассчитываемых значений beta (инициирующие значения равны истинным, чтобы избежать смещения)
    betas_matrix = array([[1, 0.5]])
    
    # Зададим список для рассчитваемых дисперсий коэффициентов
    var_b1 = []
    var_b2 = []
    true_var_b1 = []
    true_var_b2 = []
    # Зададим списки для рассчитываемых дисперсий ошибок и остатков
    var_eps = []
    var_eps_corrected = []
    true_var_eps = []
    # Счётчик для подсчёта частоты отвержения нулевой гипотезы
    t_counter = 0
    for iter in range(iterations):
        # file.write('Итерация номер: ', (iter + 1))
        
        # Генерация нормальных ошибок
        eps = normal(0, 1, (var_lengths[length_num], 1))
        
        # Генерация зависимой переменной
        y = b1 * const + b2 * x + eps

        # Расчёт OLS-оценки
        betas = dot(inv(dot(X.T, X)), dot(X.T, y))
        betas_matrix = vstack((betas_matrix, betas.T))
        
        # Оценка зависимой переменной
        y_hat = dot(X, betas)
        y_mean = full((var_lengths[length_num], 1), mean(y))
        
        # Оценка дисперсии ошибок
        var_eps.append(dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num]))
        var_eps_corrected.append(dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num] - 2))
        true_var_eps.append(dot(eps.T, (eps)) / (var_lengths[length_num]))

        # Covariance matrix under homoskedasticity: V_b = (XTX)^-1*sigma^2_e
        # Оценка истинной дисперсии
        true_var_betas = inv(dot(X.T, X)) * dot(eps.T, (eps)) / (var_lengths[length_num] - 2)
        true_var_b1.append(true_var_betas[0, 0])
        true_var_b2.append(true_var_betas[1, 1])
        # Оценка дисперсии
        var_betas = inv(dot(X.T, X)) * dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num] - 2)
        var_b1.append(var_betas[0, 0])
        var_b2.append(var_betas[1, 1])

        # Расчёт t-статистики (для b2)
        t_stat = (betas[1, 0] - 0.5) / (var_b2[iter] ** 0.5)
        if math.fabs(t_stat) > t_crits[length_num]:
            t_counter += 1
    # Рассчитаем средние оценки коэффициентов
    b1_mean = mean(betas_matrix[:, 0])
    b2_mean = mean(betas_matrix[:, 1])
    # Оценки средних дисперсий
    var_b1_mean = mean(var_b1)
    var_b2_mean = mean(var_b2)
    # Оценка средних истинных дисперсий
    true_var_b1_mean = mean(true_var_b1)
    true_var_b2_mean = mean(true_var_b2)
    # Оценка средних дисперсий ошибок
    var_eps_mean = mean(var_eps)
    var_eps_corrected_mean = mean(var_eps_corrected)
    true_var_eps_mean = mean(true_var_eps)
    file.write('\nКонец генерации одной модели с ' + str(var_lengths[length_num]) + ' наблюдениями на ' + str(iterations) + ' итерациях\n')
    file.write('Доля отвержения нулевой гипотезы на основании t-теста составила ' + str(t_counter / iterations) + '\n')
    file.write('Среднее значение для b1: ' + str(b1_mean) + '\n')
    file.write('Среднее значение для b2: ' + str(b2_mean) + '\n')
    file.write('Среднее значение дисперсии b1 составило: ' + str(var_b1_mean) + '\n')
    file.write('Среднее значение дисперсии b2 составило: ' + str(var_b2_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b1 составило: ' + str(true_var_b1_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b2 составило: ' + str(true_var_b2_mean) + '\n')
    file.write('Среднее значение дисперсии остатков составило: ' + str(var_eps_mean) + '\n')
    file.write('Среднее скорректированное значение дисперсии остатков составило: ' + str(var_eps_corrected_mean) + '\n')
    file.write('Среднее значение истинной дисперсии остатков составило: ' + str(true_var_eps_mean) + '\n')

file.write('\n\n\nНачало генерации моделей со случайным x\n')
file.write('Так как генерация X производится на каждой итерации, найдём безусловное относительно X распределение t-отношения.\n')
file.write('DGP соответствует тому, что использовался в первой модели.\n')
for length_num, length in enumerate(var_lengths):
    # Зададим список для рассчитваемых дисперсий
    var_b1 = []
    var_b2 = []
    true_var_b1 = []
    true_var_b2 = []
    # Зададим списки для рассчитываемых дисперсий ошибок и остатков
    var_eps = []
    var_eps_corrected = []
    true_var_eps = []
    # Счётчик для подсчёта частоты отвержения нулевой гипотезы
    t_counter = 0
    betas_matrix = array([[1, 0.5]])
    for iter in range(iterations):
        # file.write('Итерация номер: ' + str(iter + 1))
        # Генерация переменных, необходимых для генерации x
        ita = normal(0, 1, size=(var_lengths[length_num], 1))
        r = array([[fi ** _] for _ in range(var_lengths[length_num])])
        x0 = normal(C / (1 - fi), 1 / (1 - fi ** 2))
        A = eye(var_lengths[length_num])
        for k in range(1, var_lengths[length_num]):
            for i in range(k, var_lengths[length_num]):
                A[i][k - 1] = fi ** (i - (k - 1))
        # Генерация d
        d = ones((var_lengths[length_num], 1))
        for _ in range(var_lengths[length_num]):
            for __ in range(_):
                d[_][0] += fi ** (__ + 1)
        
        # Генерация x
        x = r * x0 + d + dot(A, ita)
        const = ones((var_lengths[length_num], 1))
        
        # Матрица значений X
        X = hstack((const, x))
        
        # Генерация нормальных ошибок
        eps = normal(0, 1, size=(var_lengths[length_num], 1))
        
        # Генерация зависимой переменной
        y = b1 * const + b2 * x + eps
        
        # Расчёт OLS-оценки
        betas = dot(inv(dot(X.T, X)), dot(X.T, y))
        betas_matrix = vstack((betas_matrix, betas.T))

        # Оценка зависимой переменной
        y_hat = dot(X, betas)
        y_mean = full((var_lengths[length_num], 1), mean(y))

        # Оценка дисперсии ошибок
        var_eps.append(dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num]))
        var_eps_corrected.append(dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num] - 2))
        true_var_eps.append(dot(eps.T, (eps)) / (var_lengths[length_num]))

        # Оценка истинной дисперсии
        true_var_betas = inv(dot(X.T, X)) * dot(eps.T, (eps)) / (var_lengths[length_num] - 2)
        true_var_b1.append(true_var_betas[0, 0])
        true_var_b2.append(true_var_betas[1, 1])

        var_betas = inv(dot(X.T, X)) * dot((y_hat - y).T, (y_hat - y)) / (var_lengths[length_num] - 2)
        var_b1.append(var_betas[0, 0])
        var_b2.append(var_betas[1, 1])
        t_stat = (betas[1, 0] - 0.5) / (var_b2[iter] ** 0.5)
        if math.fabs(t_stat) > t_crits[length_num]:
            t_counter += 1
    # Рассчитаем средние оценки коэффициентов
    b1_mean = mean(betas_matrix[:, 0])
    b2_mean = mean(betas_matrix[:, 1])
    # Оценки средних дисперсий
    var_b1_mean = mean(var_b1)
    var_b2_mean = mean(var_b2)
    # Оценка средних истинных дисперсий
    true_var_b1_mean = mean(true_var_b1)
    true_var_b2_mean = mean(true_var_b2)
    # Оценка средних дисперсий ошибок
    var_eps_mean = mean(var_eps)
    var_eps_corrected_mean = mean(var_eps_corrected)
    true_var_eps_mean = mean(true_var_eps)
    file.write('\nКонец генерации одной модели с ' + str(var_lengths[length_num]) + ' наблюдениями на ' + str(iterations) + ' итерациях\n')
    file.write('Доля отвержения нулевой гипотезы на основании t-теста составила ' + str(t_counter / iterations) + '\n')
    file.write('Среднее значение для b1: ' + str(b1_mean) + '\n')
    file.write('Среднее значение для b2: ' + str(b2_mean))
    file.write('Среднее значение дисперсии b1 составило: ' + str(var_b1_mean) + '\n')
    file.write('Среднее значение дисперсии b2 составило: ' + str(var_b2_mean))
    file.write('Среднее значение истинной дисперсии b1 составило: ' + str(true_var_b1_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b2 составило: ' + str(true_var_b2_mean) + '\n')
    file.write('Среднее значение дисперсии остатков составило: ' + str(var_eps_mean) + '\n')
    file.write('Среднее скорректированное значение дисперсии остатков составило: ' + str(var_eps_corrected_mean) + '\n')
    file.write('Среднее значение истинной дисперсии остатков составило: ' + str(true_var_eps_mean) + '\n')
file.write('\nКонец генерации моделей со случайным x')
file.write('Результаты показывают, что теоретические прогнозы сбываются: ')
file.write('1. Среднее OLS-оценок b из первого эксперимента неограниченно приближается к истинному значению (1, 0.5)\n')
file.write('2. Частота отвержения нулевой гипотезы H0 (что есть ошибка первого рода) неограниченно приближается к 5% в обоих экспериментах\n')

# Задание на стр 212
file.write('\n\nПроведём расчёты, необходимые для доказательств\n\n')
'''
-------------------------------------------------------------------------------
Надо переделать доказательства
Предположение 2.1: уравнение регрессии имеет линейный вид в силу спецификации генерации данных (y генерировалась через линейную функцию от b, X, e)
Предположение 2.2 требует, чтобы процесс {yi, xi} был стационарным и эргодическим. Для этого требуется доказать, что эти переменные i.i.d.
Для этого проведём тест Харке-Бера для переменных yi и xi, проверим их на нормальность. Тест показал неотвержение гипотезы о нормальности.
Так как у переменных распределение нормальное, то для доказательства их независимости достаточно проверить их некоррелированность.
Предположение 2.3: смешанный момент регрессоров и остатков должен иметь нулевое математическое ожидание. Эмпирические результаты подтверждаю
это предположение. В среднем смешанный момент оказался на уровне -0.00304
Предположение 2.4: матрица E(xxt) не должна быть вырожденной (не подтверждается).
Предположение 2.5: матрица перекрестных моментов E(ggt) вырождена.
Оценка дисперсий остатков и ошибок показала, что, как очевидно по формуле, дисперсия1
всегда меньше дисперсии2. Примечательно, что в соответствии с теорией, оценка дисперсии2
более близка к истинной и во всех результатах оказалась существенно точнее.
-------------------------------------------------------------------------------
'''
'''
jb_x = []
jb_y = []
cor_ar = []
t_cor_ar = []
e_x_e = []
det_xxt = []
det_ggt = []
for c, value in enumerate(var_lengths):
    t_counter = 0
    for iter in range(iterations):
        # file.write('Итерация номер: ' + str(iter + 1))
        # Генерация переменных, необходимых для генерации x
        ita = normal(0, 1, size=(var_lengths[c], 1))
        r = array([[fi ** _] for _ in range(var_lengths[c])])
        x0 = normal(C / (1 - fi), 1 / (1 - fi ** 2))
        A = eye(var_lengths[c])
        for k in range(1, var_lengths[c]):
            for i in range(k, var_lengths[c]):
                A[i][k - 1] = fi ** (i - (k - 1))
        # Генерация d
        d = ones((var_lengths[c], 1))
        for _ in range(var_lengths[c]):
            for __ in range(_):
                d[_][0] += fi ** (__ + 1)
        
        # Генерация x
        x = r*x0 + d + dot(A, ita)
        const = ones((var_lengths[c], 1))
        # Матрица значений X
        X = hstack((const, x))
        
        # Генерация нормальных ошибок
        eps = uniform(low=-0.5, high=0.5, size=(var_lengths[c], 1))
        
        # Генерация зависимой переменной
        y = b1 * const + b2 * x + eps
        
        # Расчёт OLS-оценки
        betas = dot(inv(dot(X.T, X)), dot(X.T, y))
        betas_matrix = vstack((betas_matrix, betas.T))
        y_hat = dot(X, betas)
        y_mean = array([[mean(y)] for _ in range(var_lengths[c])])
        tss = (y - y_mean) ** 2
        rss = (y_hat - y_mean) ** 2
        ess = (eps ** 2) / (var_lengths[c] - 2)
        
        sigma2 = dot(eps.T, eps) / (var_lengths[c] - 2)
        var_betas = inv(dot(X.T, X)) * sigma2
        t_stat = (betas[1][0] - 0.5) / (var_betas[1][1] ** 0.5)
        # Проверяем переменные на нормальность
        corr_x_y = hstack((x, y))
        if var_lengths[c] == 200:
            jb_x.append(jarque_bera(x[1]))  # Проверка на нормальность
            jb_y.append(jarque_bera(y[1]))  # Проверка на нормальность
            cor_ar.append(corrcoef(corr_x_y, rowvar=False)[0][1])    # Поиск корреляции
            t_cor_ar.append(cor_ar[iter] * (math.sqrt(198)) / math.sqrt(1 - cor_ar[iter]))
            x_e = x * eps               # Поиск смешанного момента
            e_x_e.append(mean(x_e))  # Поиск смешанного момента
            xxt = X.dot(X.T)# Расчёт матрицы XX'
            det_xxt.append(det(xxt))
            ggt = x_e.dot(x_e.T)    # Поиск определителя ggt
            det_ggt.append(det(ggt))
jb_x = array(jb_x)
jb_y = array(jb_y)
jb_x = mean(jb_x)
jb_y = mean(jb_y)
if jb_x > 0.05 and jb_y > 0.05:
    file.write('Гипотеза о нормальности распределения указанных величин не отвергается')
else:
    file.write('Гипотеза о нормальности распределения указанных величин отвергается')
cor_ar = array(cor_ar)
cor = mean(cor_ar)
t_cor = array(t_cor_ar)
t_cor = mean(t_cor_ar)
file.write('Значимость корреляции: ' + str(t_cor))
file.write('Среднее значением смешанного момента по всем построенным моделям: ' + str(mean(e_x_e)))
det_xxt = array(det_xxt)
file.write('Средний определитель матрицы xxt: ' + str(mean(det_xxt)))
# ---------------------------------------------------------------------------
'''
file.write('\n\n\nЗадание 4. Генерация моделей с коррекцией ошибок на некоторую сигму для создания гетероскедастичности.\n')
file.write('Проведём эксперименты Монте-Карло с двумя регрессорами, равномерно распределёнными как U[0, 1].\n')
file.write('Ошибки DGP рассчитываются как ei*sigma, где ei распределено как N(0, 1), а sigma - exp(a*x1 + a*x2^2).\n')
file.write('a - произвольная константа. Степень гетероскедастичности измеряется как lambda = max(sigma^2)/min(sigma^2).\n')
file.write('Рассчитаем оценки коэффициентов и проверим их несмещённость в условиях гетероскедастичности.\n')
file.write('Рассчитаем обычные оценки дисперсий и оценки, скорректированные на гетероскедастичность, сравним их.\n')
file.write('Рассчитаем t-статистики с использованием различных дисперсий, оценим частоту отвержения нулевой гипотезы.\n')

def XDX_matrix(X, eps):
    # Для HC0 и HC1
    xe2 = 0
    eps = diagonal(dot(eps, eps.T))
    for row in range(len(X)):
        xe2 += dot(X[row].reshape(2, 1), X[row].reshape(1, 2)) * eps[row]
    return xe2


def XDX_matrix_2(X, eps):
    # Для HC2
    # Оценка (1-h)eps
    h = diagonal(dot(dot(X, inv(dot(X.T, X))), X.T))
    eps = diagonal(dot(eps, eps.T))
    xe2 = 0
    for row in range(len(X)):
        xe2 += dot((1 - h[row]) ** -1 * X[row].reshape(2, 1), X[row].reshape(1, 2)) * eps[row]
    return xe2


def XDX_matrix_3(X, eps):
    # Для HC3
    # Оценка (1-h)eps
    h = diagonal(dot(dot(X, inv(dot(X.T, X))), X.T))
    eps = diagonal(dot(eps, eps.T))
    xe2 = 0
    for row in range(len(X)):
        xe2 += dot((1 - h[row]) ** -2 * X[row].reshape(2, 1), X[row].reshape(1, 2)) * eps[row]
    return xe2


for length_num, length in enumerate(var_lengths):
    # Зададим списки для рассчитваемых дисперсий
    var_b1, var_b2 = [], []
    var_b1_corrected, var_b2_corrected = [], []
    true_var_b1, true_var_b2 = [], []
    HC0_b1, HC0_b2 = [], []
    HC1_b1, HC1_b2 = [], []
    HC2_b1, HC2_b2 = [], []
    HC3_b1, HC3_b2 = [], []
    # Зададим счётчики отвержения нулевой гипотезы с использованием различных дисперсий
    var_b1_counter, var_b2_counter, var_b1_corrected_counter, var_b2_corrected_counter, true_var_b1_counter = 0, 0, 0, 0, 0
    true_var_b2_counter, HC0_b1_counter, HC0_b2_counter, HC1_b1_counter, HC1_b2_counter = 0, 0, 0, 0, 0
    HC2_b1_counter, HC2_b2_counter, HC3_b1_counter, HC3_b2_counter = 0, 0, 0, 0
    # Массив для рассчитываемых оценок
    betas_matrix = array([[1, 1]])
    for iter in range(iterations):
        # Коэффициенты
        b1 = 1
        b2 = 1
        # Генерация иксов
        x1 = uniform(low=0, high=1, size=(var_lengths[length_num], 1))
        x2 = uniform(low=0, high=1, size=(var_lengths[length_num], 1))
        a1 = 0.2
        # a2 = 0.4
        # Генерация сигмы для коррекции истинных ошибок в соответствии с заданием
        sigma = exp(a1 * x1 + a1 * x2 ** 2)
        eps = normal(0, 1, size=(var_lengths[length_num], 1))
        eps_cor = eps * sigma
        # Генерация истинных y
        y = b1 * x1 + b2 * x2 + eps_cor
        # Объединение иксов в матрицу
        X = hstack((x1, x2))
        # Оценка бет
        betas = dot(inv(dot(X.T, X)), dot(X.T, y))
        # Присоединение оценок к массиву
        betas_matrix = vstack((betas_matrix, betas.T))
        # Оценка зависимой переменной
        y_hat = dot(X, betas)
        # Вектор средних значений зависимой переменной
        y_mean = full((var_lengths[length_num], 1), mean(y))
        
        # ------------------Оценка дисперсий остатков и ошибок----------------------------------
        # Covariance matrix under homoskedasticity: V_b = (XTX)^-1*sigma^2_e
        var_betas = inv(dot(X.T, X)) * (dot((y - y_hat).T, (y - y_hat)) / var_lengths[length_num])
        var_b1.append(var_betas[0, 0])
        var_b2.append(var_betas[1, 1])
        var_betas_corrected = inv(dot(X.T, X)) * dot((y - y_hat).T, (y - y_hat)) / (var_lengths[length_num] - 2)
        var_b1_corrected.append(var_betas_corrected[0, 0])
        var_b2_corrected.append(var_betas_corrected[1, 1])
        # HC-оценки дисперсий
        # Covariance matrix for biased under heteroskedasricity estimator:
        # General form of the Cov matrix: (XTX)^-1*(XT*D*X)*(XTX)^-1, where
        # D = diag(s1^1, ..., sn^2) = E(e*eT|X) = E(D_|X), D_ - an unbiased estimator for D
        # eps~ = M* * eps^ where M* - diagonal matrix with i-th elements (1 - htt)^-1, M = I - X(XTX)^-1XT
        HC0 = dot(dot(inv(dot(X.T, X)), XDX_matrix(X, y - y_hat)), inv(dot(X.T, X)))
        HC0_b1.append(HC0[0, 0])
        HC0_b2.append(HC0[1, 1])
        HC1 = dot(dot(inv(dot(X.T, X)), XDX_matrix(X, y - y_hat)), inv(dot(X.T, X))) * (var_lengths[length_num] / (var_lengths[length_num] - 2))
        HC1_b1.append(HC1[0, 0])
        HC1_b2.append(HC1[1, 1])
        HC2 = dot(dot(inv(dot(X.T, X)), XDX_matrix_2(X, y - y_hat)), inv(dot(X.T, X)))
        HC2_b1.append(HC2[0, 0])
        HC2_b2.append(HC2[1, 1])
        HC3 = dot(dot(inv(dot(X.T, X)), XDX_matrix_3(X, y - y_hat)), inv(dot(X.T, X)))
        HC3_b1.append(HC3[0, 0])
        HC3_b2.append(HC3[1, 1])
        # Дисперсия на исходных данных
        true_var_betas = inv(dot(X.T, X)) * (dot(eps_cor.T, eps_cor) / var_lengths[length_num])
        true_var_b1.append(true_var_betas[0, 0])
        true_var_b2.append(true_var_betas[1, 1])

        # Оценка степени гетероскедастичности
        eet = dot((y - y_hat), (y - y_hat).T)
        eet = diagonal(eet)
        eet_max = max(eet)
        eet_min = min(eet)
        lambda_ = eet_max / eet_min

        # --------------Расчёт t-статистик----------------------------
        # t-статистики на обычных дисперсиях
        t_stat_b1 = (betas[0, 0] - 1.0) / (var_b1[iter] ** 0.5)
        if math.fabs(t_stat_b1) > t_crits[length_num]:
            var_b1_counter += 1
        t_stat_b2 = (betas[1, 0] - 1.0) / (var_b2[iter] ** 0.5)
        if math.fabs(t_stat_b2) > t_crits[length_num]:
            var_b2_counter += 1
        # t-статистики на скорректированных дисперсиях
        t_stat_b1_corrected = (betas[0, 0] - 1.0) / (var_b1_corrected[iter] ** 0.5)
        if math.fabs(t_stat_b1_corrected) > t_crits[length_num]:
            var_b1_corrected_counter += 1
        t_stat_b2_corrected = (betas[1, 0] - 1.0) / (var_b2_corrected[iter] ** 0.5)
        if math.fabs(t_stat_b2_corrected) > t_crits[length_num]:
            var_b2_corrected_counter += 1
        # t-статистики на истинных дисперсиях
        true_t_stat_b1 = (betas[0, 0] - 1.0) / (true_var_b1[iter] ** 0.5)
        if math.fabs(true_t_stat_b1) > t_crits[length_num]:
            true_var_b1_counter += 1
        true_t_stat_b2 = (betas[1, 0] - 1.0) / (true_var_b2[iter] ** 0.5)
        if math.fabs(true_t_stat_b2) > t_crits[length_num]:
            true_var_b2_counter += 1
        # t-статистики на HC0 дисперсиях
        HC0_t_stat_b1 = (betas[0, 0] - 1.0) / (HC0_b1[iter] ** 0.5)
        if math.fabs(HC0_t_stat_b1) > t_crits[length_num]:
            HC0_b1_counter += 1
        HC0_t_stat_b2 = (betas[1, 0] - 1.0) / (HC0_b2[iter] ** 0.5)
        if math.fabs(HC0_t_stat_b2) > t_crits[length_num]:
            HC0_b2_counter += 1
        # t-статистики на HC1 дисперсиях
        HC1_t_stat_b1 = (betas[0, 0] - 1.0) / (HC1_b1[iter] ** 0.5)
        if math.fabs(HC1_t_stat_b1) > t_crits[length_num]:
            HC1_b1_counter += 1
        HC1_t_stat_b2 = (betas[1, 0] - 1.0) / (HC1_b2[iter] ** 0.5)
        if math.fabs(HC1_t_stat_b2) > t_crits[length_num]:
            HC1_b2_counter += 1
        # t-статистики на HC2 дисперсиях
        HC2_t_stat_b1 = (betas[0, 0] - 1.0) / (HC2_b1[iter] ** 0.5)
        if math.fabs(HC2_t_stat_b1) > t_crits[length_num]:
            HC2_b1_counter += 1
        HC2_t_stat_b2 = (betas[1, 0] - 1.0) / (HC2_b2[iter] ** 0.5)
        if math.fabs(HC2_t_stat_b2) > t_crits[length_num]:
            HC2_b2_counter += 1
        # t-статистики на HC3 дисперсиях
        HC3_t_stat_b1 = (betas[0, 0] - 1.0) / (HC3_b1[iter] ** 0.5)
        if math.fabs(HC3_t_stat_b1) > t_crits[length_num]:
            HC3_b1_counter += 1
        HC3_t_stat_b2 = (betas[1, 0] - 1.0) / (HC3_b2[iter] ** 0.5)
        if math.fabs(HC3_t_stat_b2) > t_crits[length_num]:
            HC3_b2_counter += 1

    # Средние значения дисперсий по итерациям
    var_b1_mean, var_b2_mean, var_b1_corrected_mean, var_b2_corrected_mean = mean(var_b1), mean(var_b2), mean(var_b1_corrected), mean(var_b2_corrected)
    true_var_b1_mean, true_var_b2_mean = mean(true_var_b1), mean(true_var_b2)
    HC0_b1_mean, HC0_b2_mean, HC1_b1_mean, HC1_b2_mean = mean(HC0_b1), mean(HC0_b2), mean(HC1_b1), mean(HC1_b2)
    HC2_b1_mean, HC2_b2_mean, HC3_b1_mean, HC3_b2_mean = mean(HC2_b1), mean(HC2_b2), mean(HC3_b1), mean(HC3_b2)

    b1_mean = mean(betas_matrix[:, 0])
    b2_mean = mean(betas_matrix[:, 1])
    file.write('\nКонец генерации модели с ошибками, скорректированными на сигму, с ' + str(var_lengths[length_num]) + ' наблюдениями на ' + str(iterations) + ' итерациях\n')
    file.write('Истинные коэффициенты b1 и b2 равны ' + str(b1) + ' и ' + str(b2) + ' соответственно.\n')
    file.write('Среднее оценочное значение для b1: ' + str(b1_mean) + '\n')
    file.write('Среднее оценочное значение для b2: ' + str(b2_mean) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с обычной дисперсией составила ' + str(var_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с обычной дисперсией составила ' + str(var_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста со скорректированной дисперсией составила ' + str(var_b1_corrected_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста со скорректированной дисперсией составила ' + str(var_b2_corrected_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с истинной дисперсией составила ' + str(true_var_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с истинной дисперсией составила ' + str(true_var_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC0 дисперсией составила ' + str(HC0_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC0 дисперсией составила ' + str(HC0_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC1 дисперсией составила ' + str(HC1_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC1 дисперсией составила ' + str(HC1_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC2 дисперсией составила ' + str(HC2_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC2 дисперсией составила ' + str(HC2_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC3 дисперсией составила ' + str(HC3_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC3 дисперсией составила ' + str(HC3_b2_counter / iterations) + '\n')
    file.write('Среднее значение дисперсии b1: ' + str(var_b1_mean) + '\n')
    file.write('Среднее значение дисперсии b2: ' + str(var_b2_mean) + '\n')
    file.write('Среднее значение скорректированной дисперсии b1: ' + str(var_b1_corrected_mean) + '\n')
    file.write('Среднее значение скорректированной дисперсии b2: ' + str(var_b2_corrected_mean) + '\n')
    file.write('Среднее значение HC0 дисперсии b1 составило: ' + str(HC0_b1_mean) + '\n')
    file.write('Среднее значение HC0 дисперсии b2 составило: ' + str(HC0_b2_mean) + '\n')
    file.write('Среднее значение HC1 дисперсии b1 составило: ' + str(HC1_b1_mean) + '\n')
    file.write('Среднее значение HC1 дисперсии b2 составило: ' + str(HC1_b2_mean) + '\n')
    file.write('Среднее значение HC2 дисперсии b1 составило: ' + str(HC2_b1_mean) + '\n')
    file.write('Среднее значение HC2 дисперсии b2 составило: ' + str(HC2_b2_mean) + '\n')
    file.write('Среднее значение HC3 дисперсии b1 составило: ' + str(HC3_b1_mean) + '\n')
    file.write('Среднее значение HC3 дисперсии b2 составило: ' + str(HC3_b2_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b1 составило: ' + str(true_var_b1_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b2 составило: ' + str(true_var_b2_mean) + '\n')
    file.write('Оценка степени гетероскедастичности lambda составила: ' + str(lambda_) + '\n')
'''
'''
file.write('\n\n\nЗадание 5. Сгенерируем модель с выбросами Коши\n')
file.write('Рассчитаем все оценки, их смещение и частоту отвержения. За основу возьмём код предыдущего задания.\n')
file.write('Количество наблюдений будет фиксированным (200 штук). Ошибки генерирются на основе распределения Коши.\n')
file.write('Распределение ошибок соответствует ei ~ N(0, 1) + Cauchy(0, 10)\n')
file.write('Для создания выбросов в 10, 30 и 50 % случаев будет использоваться следующий подход:\n')
file.write('    1) будут сгенерированы три ряда длиной в 10 символов\n')
file.write('    2) в рядах будут содержаться 1, 3 и 5 чисел из распределения Коши, остальные - нули\n')
file.write('    3) будет создан массив ошибок со стандартным нормальным распределением\n')
file.write('    4) каждый элемент массива ошибок случайно будет увеличен на одно из чисел рядов с Коши\n')

Cauchy_row1 = [0, standard_cauchy(), 0, 0, 0, 0, 0, 0, 0, 0]
Cauchy_row2 = [0, standard_cauchy(), 0, standard_cauchy(), 0, 0, standard_cauchy(), 0, 0, 0]
Cauchy_row3 = [0, standard_cauchy(), 0, standard_cauchy(), 0, standard_cauchy(), 0, standard_cauchy(), 0, standard_cauchy()]
# Создаём списки, где будут лежать выбросы из распределения Коши
Cauchy_outliers1 = []
Cauchy_outliers2 = []
Cauchy_outliers3 = []

n = 200
for iter in range(n):
    Cauchy_outliers1.append(Cauchy_row1[random.randint(0, 9)])
    Cauchy_outliers2.append(Cauchy_row2[random.randint(0, 9)])
    Cauchy_outliers3.append(Cauchy_row3[random.randint(0, 9)])
# Создание массива из списка выбросов
Cauchy_outliers1 = array(Cauchy_outliers1).reshape((n, 1))
Cauchy_outliers2 = array(Cauchy_outliers2).reshape((n, 1))
Cauchy_outliers3 = array(Cauchy_outliers3).reshape((n, 1))

Cauchy_outliers = [Cauchy_outliers1, Cauchy_outliers2, Cauchy_outliers3]
cases = ['10%', '30%', '50%']
# Цикл для трёх случаев (10%, 30% и 50% выбросов соответственно)
for case in range(3):
    # Зададим списки для рассчитваемых дисперсий
    var_b1, var_b2 = [], []
    var_b1_corrected, var_b2_corrected = [], []
    true_var_b1, true_var_b2 = [], []
    HC0_b1, HC0_b2 = [], []
    HC1_b1, HC1_b2 = [], []
    HC2_b1, HC2_b2 = [], []
    HC3_b1, HC3_b2 = [], []
    # Зададим счётчики отвержения нулевой гипотезы с использованием различных дисперсий
    var_b1_counter, var_b2_counter, var_b1_corrected_counter, var_b2_corrected_counter, true_var_b1_counter = 0, 0, 0, 0, 0
    true_var_b2_counter, HC0_b1_counter, HC0_b2_counter, HC1_b1_counter, HC1_b2_counter = 0, 0, 0, 0, 0
    HC2_b1_counter, HC2_b2_counter, HC3_b1_counter, HC3_b2_counter = 0, 0, 0, 0
    # Массив для рассчитываемых оценок
    betas_matrix = array([[1, 1]])
    for iter in range(iterations):
        # Генерация данных
        b1 = 1
        b2 = 1
        x1 = uniform(low=0, high=1, size=(n, 1))
        x2 = uniform(low=0, high=1, size=(n, 1))
        eps = normal(0, 1, size=(n, 1))
        eps = eps + Cauchy_outliers[case]
        y = b1 * x1 + b2 * x2 + eps
        X = hstack((x1, x2))
        # Оценка коэффициентов
        betas = dot(inv(dot(X.T, X)), dot(X.T, y))
        betas_matrix = vstack((betas_matrix, betas.T))
        y_hat = dot(X, betas)
        y_mean = full((n, 1), mean(y))
        
        # ------------------Оценка дисперсий остатков и ошибок----------------------------------
        # Covariance matrix under homoskedasticity: V_b = (XTX)^-1*sigma^2_e
        var_betas = inv(dot(X.T, X)) * (dot((y - y_hat).T, (y - y_hat)) / 200)
        var_b1.append(var_betas[0, 0])
        var_b2.append(var_betas[1, 1])
        var_betas_corrected = inv(dot(X.T, X)) * dot((y - y_hat).T, (y - y_hat)) / 198
        var_b1_corrected.append(var_betas_corrected[0, 0])
        var_b2_corrected.append(var_betas_corrected[1, 1])
        # HC-оценки дисперсий
        # Covariance matrix for biased under heteroskedasricity estimator:
        # General form of the Cov matrix: (XTX)^-1*(XT*D*X)*(XTX)^-1, where
        # D = diag(s1^1, ..., sn^2) = E(e*eT|X) = E(D_|X), D_ - an unbiased estimator for D
        # eps~ = M* * eps^ where M* - diagonal matrix with i-th elements (1 - htt)^-1, M = I - X(XTX)^-1XT
        HC0 = dot(dot(inv(dot(X.T, X)), XDX_matrix(X, y - y_hat)), inv(dot(X.T, X)))
        HC0_b1.append(HC0[0, 0])
        HC0_b2.append(HC0[1, 1])
        HC1 = dot(dot(inv(dot(X.T, X)), XDX_matrix(X, y - y_hat)), inv(dot(X.T, X))) * (200 / 198)
        HC1_b1.append(HC1[0, 0])
        HC1_b2.append(HC1[1, 1])
        HC2 = dot(dot(inv(dot(X.T, X)), XDX_matrix_2(X, y - y_hat)), inv(dot(X.T, X)))
        HC2_b1.append(HC2[0, 0])
        HC2_b2.append(HC2[1, 1])
        HC3 = dot(dot(inv(dot(X.T, X)), XDX_matrix_3(X, y - y_hat)), inv(dot(X.T, X)))
        HC3_b1.append(HC3[0, 0])
        HC3_b2.append(HC3[1, 1])
        # Дисперсия на исходных данных
        true_var_betas = inv(dot(X.T, X)) * (dot(eps.T, eps) / 200)
        true_var_b1.append(true_var_betas[0, 0])
        true_var_b2.append(true_var_betas[1, 1])
        
        eet = dot((y - y_hat), (y - y_hat).T)
        eet = diagonal(eet)
        eet_max = max(eet)
        eet_min = min(eet)
        lambda_ = eet_max / eet_min

        # --------------Расчёт t-статистик----------------------------
        # t-статистики на обычных дисперсиях
        t_stat_b1 = (betas[0, 0] - 1.0) / (var_b1[iter] ** 0.5)
        if math.fabs(t_stat_b1) > 2.26:
            var_b1_counter += 1
        t_stat_b2 = (betas[1, 0] - 1.0) / (var_b2[iter] ** 0.5)
        if math.fabs(t_stat_b2) > 2.26:
            var_b2_counter += 1
        # t-статистики на скорректированных дисперсиях
        t_stat_b1_corrected = (betas[0, 0] - 1.0) / (var_b1_corrected[iter] ** 0.5)
        if math.fabs(t_stat_b1_corrected) > 2.26:
            var_b1_corrected_counter += 1
        t_stat_b2_corrected = (betas[1, 0] - 1.0) / (var_b2_corrected[iter] ** 0.5)
        if math.fabs(t_stat_b2_corrected) > 2.26:
            var_b2_corrected_counter += 1
        # t-статистики на истинных дисперсиях
        true_t_stat_b1 = (betas[0, 0] - 1.0) / (true_var_b1[iter] ** 0.5)
        if math.fabs(true_t_stat_b1) > 2.26:
            true_var_b1_counter += 1
        true_t_stat_b2 = (betas[1, 0] - 1.0) / (true_var_b2[iter] ** 0.5)
        if math.fabs(true_t_stat_b2) > 2.26:
            true_var_b2_counter += 1
        # t-статистики на HC0 дисперсиях
        HC0_t_stat_b1 = (betas[0, 0] - 1.0) / (HC0_b1[iter] ** 0.5)
        if math.fabs(HC0_t_stat_b1) > 2.26:
            HC0_b1_counter += 1
        HC0_t_stat_b2 = (betas[1, 0] - 1.0) / (HC0_b2[iter] ** 0.5)
        if math.fabs(HC0_t_stat_b2) > 2.26:
            HC0_b2_counter += 1
        # t-статистики на HC1 дисперсиях
        HC1_t_stat_b1 = (betas[0, 0] - 1.0) / (HC1_b1[iter] ** 0.5)
        if math.fabs(HC1_t_stat_b1) > 2.26:
            HC1_b1_counter += 1
        HC1_t_stat_b2 = (betas[1, 0] - 1.0) / (HC1_b2[iter] ** 0.5)
        if math.fabs(HC1_t_stat_b2) > 2.26:
            HC1_b2_counter += 1
        # t-статистики на HC2 дисперсиях
        HC2_t_stat_b1 = (betas[0, 0] - 1.0) / (HC2_b1[iter] ** 0.5)
        if math.fabs(HC2_t_stat_b1) > 2.26:
            HC2_b1_counter += 1
        HC2_t_stat_b2 = (betas[1, 0] - 1.0) / (HC2_b2[iter] ** 0.5)
        if math.fabs(HC2_t_stat_b2) > 2.26:
            HC2_b2_counter += 1
        # t-статистики на HC3 дисперсиях
        HC3_t_stat_b1 = (betas[0, 0] - 1.0) / (HC3_b1[iter] ** 0.5)
        if math.fabs(HC3_t_stat_b1) > 2.26:
            HC3_b1_counter += 1
        HC3_t_stat_b2 = (betas[1, 0] - 1.0) / (HC3_b2[iter] ** 0.5)
        if math.fabs(HC3_t_stat_b2) > 2.26:
            HC3_b2_counter += 1

    # Средние значения дисперсий по итерациям
    var_b1_mean, var_b2_mean, var_b1_corrected_mean, var_b2_corrected_mean = mean(var_b1), mean(var_b2), mean(var_b1_corrected), mean(var_b2_corrected)
    true_var_b1_mean, true_var_b2_mean = mean(true_var_b1), mean(true_var_b2)
    HC0_b1_mean, HC0_b2_mean, HC1_b1_mean, HC1_b2_mean = mean(HC0_b1), mean(HC0_b2), mean(HC1_b1), mean(HC1_b2)
    HC2_b1_mean, HC2_b2_mean, HC3_b1_mean, HC3_b2_mean = mean(HC2_b1), mean(HC2_b2), mean(HC3_b1), mean(HC3_b2)

    b1_mean = mean(betas_matrix[:, 0])
    b2_mean = mean(betas_matrix[:, 1])
    file.write('\nКонец генерации модели c ' + cases[case] + ' выбросов из распределения Коши с ' + str(n) + ' наблюдениями на ' + str(iterations) + ' итерациях\n')
    file.write('Истинные коэффициенты b1 и b2 равны ' + str(b1) + ' и ' + str(b2) + ' соответственно.\n')
    file.write('Среднее оценочное значение для b1: ' + str(b1_mean) + '\n')
    file.write('Среднее оценочное значение для b2: ' + str(b2_mean) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с обычной дисперсией составила ' + str(var_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с обычной дисперсией составила ' + str(var_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста со скорректированной дисперсией составила ' + str(var_b1_corrected_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста со скорректированной дисперсией составила ' + str(var_b2_corrected_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с истинной дисперсией составила ' + str(true_var_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с истинной дисперсией составила ' + str(true_var_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC0 дисперсией составила ' + str(HC0_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC0 дисперсией составила ' + str(HC0_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC1 дисперсией составила ' + str(HC1_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC1 дисперсией составила ' + str(HC1_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC2 дисперсией составила ' + str(HC2_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC2 дисперсией составила ' + str(HC2_b2_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b1 на основании t-теста с HC3 дисперсией составила ' + str(HC3_b1_counter / iterations) + '\n')
    file.write('Доля отвержения H0 по b2 на основании t-теста с HC3 дисперсией составила ' + str(HC3_b2_counter / iterations) + '\n')
    file.write('Среднее значение дисперсии b1: ' + str(var_b1_mean) + '\n')
    file.write('Среднее значение дисперсии b2: ' + str(var_b2_mean) + '\n')
    file.write('Среднее значение скорректированной дисперсии b1: ' + str(var_b1_corrected_mean) + '\n')
    file.write('Среднее значение скорректированной дисперсии b2: ' + str(var_b2_corrected_mean) + '\n')
    file.write('Среднее значение HC0 дисперсии b1 составило: ' + str(HC0_b1_mean) + '\n')
    file.write('Среднее значение HC0 дисперсии b2 составило: ' + str(HC0_b2_mean) + '\n')
    file.write('Среднее значение HC1 дисперсии b1 составило: ' + str(HC1_b1_mean) + '\n')
    file.write('Среднее значение HC1 дисперсии b2 составило: ' + str(HC1_b2_mean) + '\n')
    file.write('Среднее значение HC2 дисперсии b1 составило: ' + str(HC2_b1_mean) + '\n')
    file.write('Среднее значение HC2 дисперсии b2 составило: ' + str(HC2_b2_mean) + '\n')
    file.write('Среднее значение HC3 дисперсии b1 составило: ' + str(HC3_b1_mean) + '\n')
    file.write('Среднее значение HC3 дисперсии b2 составило: ' + str(HC3_b2_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b1 составило: ' + str(true_var_b1_mean) + '\n')
    file.write('Среднее значение истинной дисперсии b2 составило: ' + str(true_var_b2_mean) + '\n')
    file.write('Оценка степени гетероскедастичности lambda составила: ' + str(lambda_) + '\n')
