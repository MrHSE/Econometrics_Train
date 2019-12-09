# -*- coding: utf-8 -*-
"""
Created on Sat Okt 13 12:24:32 2019
@author: Daniil
"""
from numpy import dot, array, eye, ones, zeros, hstack, vstack, mean, transpose, corrcoef, diagonal, reshape, exp, full, diagflat, abs, split, shape
from numpy.random import normal, uniform, standard_cauchy, randint, choice
from numpy.linalg import inv, det
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from math import sin, cos
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys


def ma_q(T, z, *coefs):
    # MA(q)-процесс: yt = b1 * mu + b2 * Xt + ut, где Xt = eps + z*eps_t-1
    # Генерация Xt
    eps_xt = normal(0, 3, size=(T, 1))
    eps_xt_1 = var_shift(eps_xt, 1)
    Xt = eps_xt + z * eps_xt_1
    # Генерация константы и ошибок
    mu = ones((T, 1))
    ut = normal(0, 1, size=(T, 1))
    # Генерация yt
    yt = coefs[0] * mu + coefs[1] * Xt + ut
    # Объединение константы и зависимой переменной в матрицу
    Xt = hstack((mu, Xt))

    return yt, Xt


def ar_r(T, *coefs):
    # AR(r)-процесс: Xt = b1 * const + r1 * xt-1 + epst;
    const = ones((T, 1))
    xt_src = normal(1, 2, size=(T, 1))
    xt_1 = var_shift(xt_src, 1)
    eps = normal(0, 1, size=(T, 1))
    
    # Генерация Xt
    Xt = coefs[0] * const + coefs[1] * xt_1 + eps
    # Объединение константы и зависимой переменной в матрицу
    Xt_lags = hstack((const, xt_1))

    return Xt, Xt_lags


def XDX_bartlett_matrix(X, eps, lag):
    # Для ковариационной матрицы ошибок для расчёта HAC-оценки
    # L - максимально дальний элемент ковариационной матрицы от диагонали
    L = int(4 * ((len(X) / 100)) ** (2/9))
    xe2 = 0
    for row in range(len(X)):
        # Часть e^2 * x * xT
        xe2 += dot(X[row].reshape(2, 1), X[row].reshape(1, 2)) * eps[row] ** 2
    # Часть Ковариационной матрицы для оценки внедиагональных элементов ядром Бартлетта
    for diag in range(L):
        for row in range(lag, len(X)):
            x = abs(lag / (diag + 1))
            xe2 += (1 - x) * eps[row] * eps[row - lag] * (dot(X[row].reshape(2,1), X[row - lag].reshape(1,2)) + dot(X[row - lag].reshape(2,1), X[row].reshape(1,2)))
    return xe2


def XDX_QS_matrix(X, eps, lag):
    # Для ковариационной матрицы ошибок для расчёта HAC-оценки
    # L - максимально дальний элемент ковариационной матрицы от диагонали
    L = int(4 * ((len(X) / 100)) ** (2/9))
    xe2 = 0
    pi = 3.1416
    for row in range(len(X)):
        # Часть e^2 * x * xT
        xe2 += dot(X[row].reshape(2, 1), X[row].reshape(1, 2)) * eps[row] ** 2
    # Часть Ковариационной матрицы для оценки внедиагональных элементов c квадратичным спектральным ядром
    for diag in range(L):
        for row in range(lag, len(X)):
            x = lag / (diag + 1)
            xe2 += ((25/(12*pi**2*x**2)) * (sin(6*pi*x/5) / (6*pi*x/5) - cos(6*pi*x/5))) * eps[row] * eps[row - lag] * (dot(X[row].reshape(2,1), X[row - lag].reshape(1,2)) + dot(X[row - lag].reshape(2,1), X[row].reshape(1,2)))
    return xe2


def var_shift(var, shift):
    # Функция, чтобы получить ряд xt-1
    new_var = zeros(shape=(len(var), 1))
    new_var[shift:, 0] = var[:len(var) - shift, 0]
    return new_var


def table_part ():
    # Часть 1
    T = 128
    coefs_vars = [-0.5, 0.5, 0.9, 0.95]
    groups = [2, 4, 8, 16]

    total_t_ar = []
    total_t_ma = []
    total_t_ar_corr = []
    total_t_ma_corr = []
    total_t = []
    B = int(input("Определите количество итераций: "))
    for coef_num, coef in enumerate(coefs_vars):
        print(str(coef_num / 4 * 100) + '% выполнено')
        # Счётчики для количества отвержений t-статистики
        h0_rej_ma_gr2, h0_rej_ma_gr4, h0_rej_ma_gr8, h0_rej_ma_gr16, h0_rej_ma_b, h0_rej_ma_qs = 0, 0, 0, 0, 0, 0
        h0_rejection_ma_list = [h0_rej_ma_gr2, h0_rej_ma_gr4, h0_rej_ma_gr8, h0_rej_ma_gr16, h0_rej_ma_b, h0_rej_ma_qs]
        h0_rej_ar_gr2, h0_rej_ar_gr4, h0_rej_ar_gr8, h0_rej_ar_gr16, h0_rej_ar_b, h0_rej_ar_qs = 0, 0, 0, 0, 0, 0
        h0_rejection_ar_list = [h0_rej_ar_gr2, h0_rej_ar_gr4, h0_rej_ar_gr8, h0_rej_ar_gr16, h0_rej_ar_b, h0_rej_ar_qs]
        # Счётчики для скорректированных статистики
        h0_rej_ma_gr2_corr, h0_rej_ma_gr4_corr, h0_rej_ma_gr8_corr, h0_rej_ma_gr16_corr, h0_rej_ma_b_corr, h0_rej_ma_qs_corr = 0, 0, 0, 0, 0, 0
        h0_rejection_ma_corr_list = [h0_rej_ma_gr2_corr, h0_rej_ma_gr4_corr, h0_rej_ma_gr8_corr, h0_rej_ma_gr16_corr, h0_rej_ma_b_corr, h0_rej_ma_qs_corr]
        h0_rej_ar_gr2_corr, h0_rej_ar_gr4_corr, h0_rej_ar_gr8_corr, h0_rej_ar_gr16_corr, h0_rej_ar_b_corr, h0_rej_ar_qs_corr = 0, 0, 0, 0, 0, 0
        h0_rejection_ar_corr_list = [h0_rej_ar_gr2_corr, h0_rej_ar_gr4_corr, h0_rej_ar_gr8_corr, h0_rej_ar_gr16_corr, h0_rej_ar_b_corr, h0_rej_ar_qs_corr]
        for _ in range(B):
            # -----------------Оценка t-статистики c использованием оценки Нью-Веста------------------------------------
            # Генерация MA(1)-процесса по заданной функции
            b1, z = 2, 1.5
            y_ma, Xt_ma = ma_q(T, z, b1, coef)
            # Оценка коэффициентов
            betas_ma = dot(inv(dot(Xt_ma.T, Xt_ma)), dot(Xt_ma.T, y_ma))
            y_ma_hat = dot(Xt_ma, betas_ma)
            # Оценка дисперсии по формуле Ньюи-Веста с ядром Бартлетта
            var_b_west = dot(dot(inv(dot(Xt_ma.T, Xt_ma)), XDX_bartlett_matrix(Xt_ma, y_ma - y_ma_hat, 1)), inv(dot(Xt_ma.T, Xt_ma)))
            t_stat = (betas_ma[1, 0] - coef) / (var_b_west[1, 1] ** 0.5)
            if abs(t_stat) > 1.9788:
                h0_rejection_ma_list[4] += 1
            # Скорректировано на размер
            t_stat = (betas_ma[1, 0] - coef) / (var_b_west[1, 1] ** 0.5) + 5 / 128 ** 0.5
            if abs(t_stat) > 1.9788:
                h0_rejection_ma_corr_list[4] += 1
            # Оценка дисперсии по формуле Ньюи-Веста с квадратичным спектральным ядром
            var_qs_west = dot(dot(inv(dot(Xt_ma.T, Xt_ma)), XDX_QS_matrix(Xt_ma, y_ma - y_ma_hat, 1)), inv(dot(Xt_ma.T, Xt_ma)))
            t_stat = (betas_ma[1, 0] - coef) / (var_qs_west[1, 1] ** 0.5)
            if abs(t_stat) > 1.9788:
                h0_rejection_ma_list[5] += 1
            # Скорректировано на размер
            t_stat = (betas_ma[1, 0] - coef) / (var_qs_west[1, 1] ** 0.5) + 5 / 128 ** 0.5
            if abs(t_stat) > 1.9788:
                h0_rejection_ma_corr_list[5] += 1

            # Генерация AR(1)-процесса по заданной функции
            Xt, xt_lags = ar_r(T, b1, coef)
            # Оценка коэффициентов
            betas_ar = dot(inv(dot(xt_lags.T, xt_lags)), dot(xt_lags.T, Xt))
            Xt_hat = dot(xt_lags, betas_ar)
            # Оценка дисперсии по формуле Ньюи-Веста с ядром Бартлетта
            var_b_west = dot(dot(inv(dot(xt_lags.T, xt_lags)), XDX_bartlett_matrix(xt_lags, Xt - Xt_hat, 1)), inv(dot(xt_lags.T, xt_lags)))
            t_stat = (betas_ar[1, 0] - coef) / (var_b_west[1, 1] ** 0.5)
            if abs(t_stat) > 1.9788:
                h0_rejection_ar_list[4] += 1
            # Скорректировано на размер
            t_stat = (betas_ar[1, 0] - coef) / (var_b_west[1, 1] ** 0.5) + 5 / 128 ** 0.5
            if abs(t_stat) > 1.9788:
                h0_rejection_ar_corr_list[4] += 1
            # Оценка дисперсии по формуле Ньюи-Веста с квадратичным спектральным ядром
            var_qs_west = dot(dot(inv(dot(xt_lags.T, xt_lags)), XDX_QS_matrix(xt_lags, Xt - Xt_hat, 1)), inv(dot(xt_lags.T, xt_lags)))
            t_stat = (betas_ar[1, 0] - coef) / (var_qs_west[1, 1] ** 0.5)
            if abs(t_stat) > 1.9788:
                h0_rejection_ar_list[5] += 1
            # Скорректировано на размер
            t_stat = (betas_ar[1, 0] - coef) / (var_qs_west[1, 1] ** 0.5) + 5 / 128 ** 0.5
            if abs(t_stat) > 1.9788:
                h0_rejection_ar_corr_list[5] += 1
            # --------- Цикл для оценок t-статистики с разделением выборки на группы-------------------------------------------
            for group_num, group in enumerate(groups):
                # Разделение генеральной выборки на q выборок для оценки MA(1)
                ma_Xt_arrays = split(Xt_ma, group, axis=0)
                y_ma_arrays = split(y_ma, group, axis=0)
                ma_beta_estimations = []
                ma_var_b_ests = []

                # Разделение генеральной выборки на q выборок для оценки AR(1)
                ar_xt_lags_arrays = split(xt_lags, group, axis=0)
                Xt_arrays = split(Xt, group, axis=0)
                ar_beta_estimations = []
                ar_var_b_ests = []
                for iter_ in range(group):
                    # Итерации при оценке MA(1)
                    cur_ma_Xt = ma_Xt_arrays[iter_]
                    cur_y_ma = y_ma_arrays[iter_]
                    ma_cur_betas = dot(inv(dot(cur_ma_Xt.T, cur_ma_Xt)), dot(cur_ma_Xt.T, cur_y_ma))
                    # Оценка беты
                    ma_beta_estimations.append(ma_cur_betas[1][0])
                    cur_y_ma_hat = dot(cur_ma_Xt, ma_cur_betas)
                    # cur_y_ma_mean = np.array([[np.mean(cur_y_ma)] for _ in range(len(ma_cur_x))])
                    ma_var_betas = inv(dot(transpose(cur_ma_Xt), cur_ma_Xt)) * dot(transpose(cur_y_ma_hat - cur_y_ma), cur_y_ma_hat - cur_y_ma) / (len(cur_ma_Xt) - 2)
                    ma_var_b_ests.append(ma_var_betas[1][1])

                    # Итерации при оценке AR(1)
                    ar_cur_Xt_lags = ar_xt_lags_arrays[iter_]
                    ar_cur_Xt = Xt_arrays[iter_]
                    ar_cur_betas = dot(inv(dot(transpose(ar_cur_Xt_lags), ar_cur_Xt_lags)), dot(transpose(ar_cur_Xt_lags), ar_cur_Xt))
                    ar_beta_estimations.append(ar_cur_betas[1][0])
                    cur_Xt_hat = dot(ar_cur_Xt_lags, ar_cur_betas)
                    # cur_Xt_mean = np.array([[np.mean(ar_cur_Xt)] for _ in range(len(ar_cur_x))])
                    ar_var_betas = inv(dot(transpose(ar_cur_Xt_lags), ar_cur_Xt_lags)) * dot(transpose(cur_Xt_hat - ar_cur_Xt), cur_Xt_hat - ar_cur_Xt) / (len(ar_cur_Xt) - 2)
                    ar_var_b_ests.append(ar_var_betas[1][1])
                # Оценка t-статистики для MA(1)-процесса для групп q
                group_beta_mean = mean(ma_beta_estimations)
                group_var_b_estimation = sum(ma_var_b_ests) / (group - 1)
                cur_t_stat = (group ** 0.5) * (group_beta_mean - coef) / (group_var_b_estimation ** 0.5)
                if abs(cur_t_stat) > 1.9788:
                    h0_rejection_ma_list[group_num] += 1
                # Скорректировано на размер
                cur_t_stat = (group ** 0.5) * (group_beta_mean - coef) / (group_var_b_estimation ** 0.5) + 5 / 128 ** 0.5
                if abs(cur_t_stat) > 1.9788:
                    h0_rejection_ma_corr_list[group_num] += 1
                
                # Оценка t-статистики для AR(1)-процесса для групп q
                group_beta_mean = mean(ar_beta_estimations)
                group_var_b_estimation = sum(ar_var_b_ests) / (group - 1)
                cur_t_stat = (group ** 0.5) * (group_beta_mean - coef) / (group_var_b_estimation ** 0.5)
                if abs(cur_t_stat) > 1.9788:
                    h0_rejection_ar_list[group_num] += 1
                # Скорректировано на размер
                cur_t_stat = (group ** 0.5) * (group_beta_mean - coef) / (group_var_b_estimation ** 0.5) + 4 / (128 * (1 - coef ** 2)) ** 0.5
                if abs(cur_t_stat) > 1.9788:
                    h0_rejection_ar_corr_list[group_num] += 1
        for counter_num, _ in enumerate(h0_rejection_ar_list):
            h0_rejection_ar_list[counter_num] = round(h0_rejection_ar_list[counter_num] / B * 100, 1)
            h0_rejection_ma_list[counter_num] = round(h0_rejection_ma_list[counter_num] / B * 100, 1)
            h0_rejection_ar_corr_list[counter_num] = round(h0_rejection_ar_corr_list[counter_num] / B * 100, 1)
            h0_rejection_ma_corr_list[counter_num] = round(h0_rejection_ma_corr_list[counter_num] / B * 100, 1)
        h0_rejection_ar_list.insert(0, 'AR(1) ' + str(coef))
        h0_rejection_ma_list.insert(0, 'MA(1) ' + str(coef))
        h0_rejection_ar_corr_list.insert(0, 'AR(1)-corrected ' + str(coef))
        h0_rejection_ma_corr_list.insert(0, 'MA(1)-corrected ' + str(coef))

        total_t_ma.append(h0_rejection_ma_list)
        total_t_ar.append(h0_rejection_ar_list)
        total_t_ma_corr.append(h0_rejection_ma_corr_list)
        total_t_ar_corr.append(h0_rejection_ar_corr_list)

    # Массив из массивов массивов оценок t-статистик для MA- и AR-процессов
    print('100% выполнено')
    total_t.append(total_t_ma)
    total_t.append(total_t_ar)
    total_t.append(total_t_ma_corr)
    total_t.append(total_t_ar_corr)
    table = PrettyTable()
    table.field_names = ['Процесс', 'q(2)', 'q(4)', 'q(8)', 'q(16)', 'Newey-West_Bartlett', 'Newey-West_QS']

    # Заполнение таблицы результатами оценки
    for proc in total_t:
        for res in proc:
            table.add_row(res)
    print(table)



def financial_research():
    # Часть 2
    os.chdir('/home/alex/Programs/Python/Econometrics')
    # Данные представлены недельными ценовыми наблюдениями акций компании ПАО Алроса за период 
    # с 28.11.11 по 26.02.18
    data = pd.read_csv('econometrics2_data.txt', sep=' ',dtype='float64')
    # print(data)
    # file = open('econometrics2_data.txt', 'r')
    # data = []
    # for line in file:
    #     data.append(line[:-1])
    # data = array(data)
    # data = data.reshape(1, len(data))
    plot_acf(x=data['Алроса'], use_vlines=True, title='ACF', unbiased=True, zero=True)
    plt.savefig('ACF')
    plot_pacf(x=data['Алроса'], use_vlines=True, title='PACF', zero=True)
    plt.savefig('PACF')
    # p_max - оценка максимального количества лагов в модели
    # T = 128
    # p_max = 4 * ((T / 100) ** (1/3))
    # print(round(p_max))
    # print(data)
    # print(var_shift(data, 1))


def res_bootstrap(X, betas_hat, eps_hat, t_crit, B):
    # Идея: ресэмлировать остатки с возвратом для генерации бутстрапированных моделей
    print('Расчёт бутстрапа по остаткам для', len(X), 'наблюдений')
    h0_reject_freq = 0
    for _ in range(B):
        # Случайный отбор ошибки из уже сгенерированных
        bootstrap_eps = array([[eps_hat[randint(0, len(X))][0]] for _ in range(len(X))])
        # Генерация Y с учетом новых ошибок
        Y = dot(X, betas_hat) + bootstrap_eps
        bootstrap_betas = dot(inv(dot(transpose(X), X)), dot(transpose(X), Y))
        Y_hat = dot(X, bootstrap_betas)

        var_betas = inv(dot(transpose(X), X)) * dot(transpose(Y_hat - Y), Y_hat - Y) / (len(Y) - 2)
        t_stat = abs((bootstrap_betas[1, 0] - betas_hat[1, 0]) / (var_betas[1, 1] ** 0.5))
        if t_stat > t_crit:
            h0_reject_freq += 1
    return h0_reject_freq


def par_bootstrap(X, betas_hat, eps_hat, t_crit, B):
    # Идея: генерировать ошибки по распределению остатков
    print('Расчёт параметрического бутстрапа для', len(X), 'наблюдений')
    h0_reject_freq = 0
    for _ in range(B):
        # Ошибки распределены нормально (т.к таков DGP), оценим параметры распределения
        eps_mean = mean(eps_hat)
        eps_sigma = dot(transpose(eps_hat), eps_hat) / (len(eps_hat) - 2)
        # Генерация ошибок по выборочным параметрам
        bootstrap_eps = normal(eps_mean, eps_sigma, size=(len(eps_hat), 1))
        # Генерация Y с учетом новых ошибок
        Y = dot(X, betas_hat) + bootstrap_eps
        bootstrap_betas = dot(inv(dot(transpose(X), X)), dot(transpose(X), Y))
        Y_hat = dot(X, bootstrap_betas)

        var_betas = inv(dot(transpose(X), X)) * dot(transpose(Y_hat - Y), Y_hat - Y) / (len(X) - 2)
        t_stat = abs((bootstrap_betas[1, 0] - betas_hat[1, 0]) / (var_betas[1, 1] ** 0.5))
        if t_stat > t_crit:
            h0_reject_freq += 1
    return h0_reject_freq


def wild_bootstrap(X, betas_hat, eps, t_crit, B):
    # Wild bootstrap нужен, чтобы контролировать гетероскедастичность в выборке
    # Идея: генерация ошибок на основе оценки дисперсии Уайта для контроля гетероскедастичности
    print('Расчёт wild бутстрапа для', len(X), 'наблюдений')
    # Оценка (1-h)eps
    h = diagonal(dot(dot(X, inv(dot(transpose(X), X))), transpose(X)))
    for row in range(len(X)):
        eps[row, 0] = (1 - h[row]) ** -0.5 * eps[row, 0]
    h0_reject_freq = 0
    for _ in range(B):
        # Генерация случайной ошибки wild-бутстрапа
        # nu = normal(0, 1, size=(len(X), 1))
        nu = array([[choice([-1, 1])] for _ in range(len(X))]).reshape(len(X), 1)
        eps = nu * eps
        # Генерация Y с учетом новых ошибок
        Y = dot(X, betas_hat) + eps
        bootstrap_betas = dot(inv(dot(X.T, X)), dot(X.T, Y))
        Y_hat = dot(X, bootstrap_betas)

        var_betas = inv(dot(transpose(X), X)) * dot((Y_hat - Y).T, Y_hat - Y) / (len(X) - 2)
        t_stat = abs((bootstrap_betas[1, 0] - betas_hat[1, 0]) / (var_betas[1, 1] ** 0.5))
        if t_stat > t_crit:
            h0_reject_freq += 1
    return h0_reject_freq


def pair_bootstrap(X, Y, betas_hat, t_crit, B):
    # Вроде работает даже в случае гетероскедастичности ошибок
    # И для временных рядов
    print('Расчёт парного бутстрапа для', len(X), 'наблюдений')
    h0_reject_freq = 0
    for _ in range(B):
        observations = [i for i in range(len(X))]
        new_observations = [choice(observations) for _ in range(len(X))]
        # Инициализация массива с новыми значениями независимых переменных
        X_new = X[new_observations[0], :]
        Y_new = Y[new_observations[0], :]
        for new_obs_num in range(1, len(X)):
            X_new = vstack((X_new, X[new_observations[new_obs_num], :]))
            Y_new = vstack((Y_new, Y[new_observations[new_obs_num], :]))
        bootstrap_betas = dot(inv(dot(transpose(X_new), X_new)), dot(transpose(X_new), Y_new))
        Y_hat = dot(X_new, bootstrap_betas)

        var_betas = inv(dot(transpose(X_new), X_new)) * dot(transpose(Y_hat - Y_new), Y_hat - Y_new) / (len(X) - 2)
        t_stat = abs((bootstrap_betas[1, 0] - betas_hat[1, 0]) / (var_betas[1, 1] ** 0.5))
        if t_stat > t_crit:
            h0_reject_freq += 1
    return h0_reject_freq


def student_res(yt_mean, yt_sigma, ut_sigma, b1, b2, length, t_crit, B):
    h0_reject_freq = 0
    for _ in range(B):
        yt = normal(yt_mean, yt_sigma, size=(length, 1))
        yt_1 = var_shift(yt, 1)
        const = full((length, 1), 1)
        ut = normal(0, ut_sigma, size=(length, 1))
        
        # Генерация зависимой переменной
        Y = b1 * const + b2 * yt_1 + ut
        # Расчёт OLS-оценки
        X = hstack((const, yt_1))
        betas_hat = dot(inv(dot(transpose(X), X)), dot(transpose(X), Y))

        # Оценка зависимой переменной
        Y_hat = dot(X, betas_hat)

        # Оценка остатков
        eps_hat = Y - Y_hat
        # Оценка t-статистики
        var_betas = inv(dot(X.T, X)) * dot(eps_hat.T, eps_hat) / (len(X) - 2)
        t_stat = abs(betas_hat[1, 0] - 5) / var_betas[1, 1] ** 0.5
        if t_stat > t_crit:
            h0_reject_freq += 1
    return h0_reject_freq


def bootstrap_part():
    # Часть 3
    # Количество наблюдений в различных генерациях
    # Список количества наблюдений для различных генераций
    var_lengths = [10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320, 452, 640, 905, 1280]
    # Критические значения t-статистики для соответствующего количества наблюдений
    t_crits = [2.306, 2.179, 2.101, 2.056, 2.024, 2.005, 1.991, 1.9816, 1.975, 1.9706, 1.9675, 1.9652, 1.9637, 1.9626, 1.9618]
    results = []
    # B = 399, согласно MacKinnon
    B = int(input("Напишите количество бутстраповских повторов: "))
    # Сгенерируем модель вида yt = b1 + b2*y_t-1 + ut;
    b1, b2 = 3, 5
    for length_num, length in enumerate(var_lengths):
        print('Генерация модели с ' + str(length) + ' наблюдениями')
        yt_mean = 0
        yt_sigma = 3
        yt = normal(yt_mean, yt_sigma, size=(length, 1))
        yt_1 = var_shift(yt, 1)
        const = full((length, 1), 1)
        ut_sigma = 2
        ut = normal(0, ut_sigma, size=(length, 1))

        # Генерация зависимой переменной
        Y = b1 * const + b2 * yt_1 + ut
        # Расчёт OLS-оценки
        X = hstack((const, yt_1))
        betas_hat = dot(inv(dot(transpose(X), X)), dot(transpose(X), Y))

        # Оценка зависимой переменной
        Y_hat = dot(X, betas_hat)

        # Оценка остатков
        eps_hat = Y - Y_hat
        # Rescaling residuals so that they have the correct variance
        new_eps_hat = (len(X) / (len(X) - shape(X)[1])) ** 0.5 * eps_hat

        # Расчёт бутстрапированных оценок
        boot_res_results = res_bootstrap(X, betas_hat, new_eps_hat, t_crits[length_num], B) / B
        boot_par_results = par_bootstrap(X, betas_hat, eps_hat, t_crits[length_num], B) / B
        boot_wild_results = wild_bootstrap(X, betas_hat, eps_hat, t_crits[length_num], B) / B
        boot_pair_results = pair_bootstrap(X, Y, betas_hat, t_crits[length_num], B) / B
        t_student = student_res(yt_mean, yt_sigma, ut_sigma, b1, b2, length, t_crits[length_num], B) / B
        results.append([t_student, boot_res_results, boot_par_results, boot_wild_results, boot_pair_results])

    pd.options.display.float_format = '{:,.3f}'.format
    sns.set(style='darkgrid')
    results = pd.DataFrame(data=results, columns=['Student', 'Residual', 'Parametric', 'Wild', 'Pair'], dtype='float64')
    ax = sns.lineplot(data=results)
    plt.savefig('test.png')


if __name__ == "__main__":
    financial_research()
