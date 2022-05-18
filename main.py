# В18.Лабораторная работа 3.1. Решение систем линейных алгебраических уравнений
# 1. Получить решение системы линейных алгебраических уравнений
# с точностью до четырёх верных знаков методом простой итерации
# и методом Зейделя, предварительно проверить условия сходимости
# метода. Сравнить методы по скорости сходимости.

import numpy as np
from numpy import linalg


def SIM(uB, ub, e):
    n = len(uB[0][:])
    rez = ub
    rez1 = np.empty((n, 1))
    rez1[:] = np.NAN
    k = 0
    while k == 0 or (np.max(np.abs(rez - rez1))) > e:
        rez1 = rez
        rez = uB * rez1 + ub
        k = k + 1
    return [np.diag(rez), k]


def Seidel(uB, ub, e):
    n = len(uB[0][:])
    H = np.tril(uB, -1)
    F = np.triu(uB)
    eh = np.linalg.inv(np.eye(n) - H)
    rez = ub
    rez1 = np.empty((n, 1))
    rez1[:] = np.NAN
    k = 0
    while k == 0 or (np.max(np.abs(rez - rez1))) > e:
        rez1 = rez
        rez = eh * F * rez1 + eh * ub
        k = k + 1
    return [np.diag(rez), k]


# 2. Получить решение системы линейных алгебраических уравнений
# методом квадратного корня.
def Msqrt(uB, ub):
    n = len(uB[0][:])

    S = np.zeros((n, n), dtype=complex)
    D = np.zeros((n, n), dtype=complex)

    D[0, 0] = np.sign(uB[0, 0])
    S[0, 0] = np.sqrt(np.abs(uB[0, 0]))
    S[0, 1:n] = uB[0, 1:n] / (D[0, 0] * S[0, 0])
    eh = 0
    for i in range(1, n):
        if i == 1:
            eh = uB[i, i] - pow((np.abs(S[0, i])), 2) * D[0, 0]
        else:
            eh = uB[i, i] - np.sum(pow((np.abs(S[0:i, i])), 2) * np.diag(D[0:i, 0:i]))

        D[i, i] = np.sign(eh)
        S[i, i] = pow(eh, 1/2)
        for j in range((i + 1), n):
            S[i, j] = (uB[i, j] - np.sum(np.conj(S[0:i, i]) * np.diag(D[0:i, 0:i]) *
                                         S[0:i, j])) / (D[i, i] * S[i, i])

    rez1 = np.zeros(n, dtype=complex)
    rez1[0] = ub[0] / S[0][0]
    for i in range(1, n):
        rez1[i] = ub[i]
        for j in range(0, i):
            rez1[i] = rez1[i] - S[j, i] * rez1[j]
        rez1[i] = rez1[i] / S[i, i]
    rez = np.zeros(n, dtype=complex)
    rez[n-1] = (rez1[n-1]) / S[n-1, n-1]
    for i in range(n-2, -1, -1):
        rez[i] = rez1[i]
        sun = 0
        j = n - 1
        while j >= i+1:
            sun = sun - S[i, j] * rez[j]
            j = j - 1
        rez[i] = (rez[i]+sun)/S[i, i]

    return rez


def main():
    B = np.array(
        [[0.17, 0.27, -0.13, -0.11], [0.13, -0.12, 0.09, -0.06], [0.11, 0.05, -0.02, 0.12], [0.13, 0.18, 0.24, 0.43]])
    b = np.array([-1.42, 0.48, -2.34, 0.72])
    eps = pow(0.1, 3)
    if np.amax((np.sum(np.abs(B), axis=1))) < 1:
        print("Достаточное условие выполнено\n")
        sim = SIM(B, b, eps)
        print("Метод простой итерации: ")
        print("Решение: \n", sim[0], "\nКоличество итераций: ", sim[1])

        print("Достаточное условие выполнено\n")
        sei = Seidel(B, b, eps)
        print("Метод Зейделя: ")
        print("Решение: \n", sei[0], "\nКоличество итераций: ", sei[1])
    else:
        print("Достаточное условие не выполнено\n")
    #############################
    B = np.array(
        [[0.75, -1.24, 1.56], [-1.24, 0.18, -1.72], [1.56, -1.72, 0.79]])
    b = np.array([0.49, -0.57, 1.03])

    print("Метод квадратного корня: ")
    print("Решение: \n", Msqrt(B, b))
    return 0


main()
