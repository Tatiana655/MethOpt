import math as m
import numpy as np


def F(x1, x2):
    return x1 ** 2 + x2 ** 2 + m.sin(2 * x1 + 3 * x2) + 3 * x1 + 2 * x2


def GradFx(x1, x2):
    return 2 * x1 + 2 * m.cos(2 * x1 + 3 * x2) + 3


def GradFy(x1, x2):
    return 2 * x2 + 3 * m.cos(2 * x1 + 3 * x2) + 2


def H11(x1, x2):
    return 2 - 4 * m.sin(2 * x1 + 3 * x2)


def H12(x1, x2):
    return -6 * m.sin(2 * x1 + 3 * x2)


def H21(x1, x2):
    return -6 * m.sin(2 * x1 + 3 * x2)


def H22(x1, x2):
    return 2 - 9 * m.sin(2 * x1 + 3 * x2)


def Grad1(eps, x_0, y_0, alpha):
    f = False
    x_old = x_0
    y_old = y_0
    x_new = x_0
    y_new = y_0
    count = 0
    while (not f) or (np.linalg.norm([GradFx(x_new, y_new), GradFy(x_new, y_new)]) > eps):
        f = True
        tmp1 = x_new
        tmp2 = y_new
        x_new = x_old - alpha * GradFx(x_old, y_old)
        y_new = y_old - alpha * GradFy(x_old, y_old)
        x_old = tmp1
        y_old = tmp2
        count += 1
        print(x_new, y_new)
        print(np.linalg.norm([GradFx(x_new, y_new), GradFy(x_new, y_new)]))
    return count


# тут не понятно что такое p_k и что за матрицу вообще брать надо
def Grad2Prop(eps, x_0, y_0, lam, delt):  # 0<lam<1, 0<delt<0.5
    f = False

    x_old = x_0
    y_old = y_0
    x_new = x_0
    y_new = y_0
    count = 0
    while (not f) or (np.linalg.norm([GradFx(x_new, y_new), GradFy(x_new, y_new)]) > eps):
        f = True
        count += 1
        alpha = 1
        A = [[H11(x_old, y_old), H12(x_old, y_old)], [H21(x_old, y_old), H22(x_old, y_old)]]
        b = [GradFx(x_old, x_old), GradFy(x_old, x_old)]
        pk = np.linalg.solve(A, b)

        while F(x_old + alpha * pk[0], y_old + alpha * pk[1]) - F(x_old, y_old) > alpha * delt * (
                GradFx(x_old, y_old) * pk[0] + GradFy(x_old, y_old) * pk[1]):
            alpha = alpha * lam

        tmp1 = x_new
        tmp2 = y_new
        x_new = x_old - alpha * pk[0]
        y_new = y_old - alpha * pk[1]
        x_old = tmp1
        y_old = tmp2
    return count




# начальне приближение# выбирала на глазок
x_0 = -2.5  # -3
y_0 = -2.5  # -2
# шаг
lam = 0.1  # 0.001 0.01
# точность относительно минимизируюшей функции
eps = 0.01
# print("итерации", Grad1(0.1, -2.5, -2.5, 0.1))
#print(Grad2Prop(eps, x_0, y_0, 0.1, 0.5))


