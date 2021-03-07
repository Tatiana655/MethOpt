import matplotlib.pyplot as plt
import numpy as np

# 1)Решить задачу методом равномерного поиска и метод золотого сечения.
# Без вычисления производных применимы к унимодальным функциям.
# Где-то на каком-то интервале, уточняем интервал с требуемой точностью.
# Графики
# 2)Сравниваются по числу обращений к функции, составить табличку с числом обращений 0.1,0.01,0.001 и число для заданной точности
# 3)Формулу связи начального и конечного интервала неопределённости для методов. после заданного числа обращений к вычислению функции
# 4)?: Что выгоднее в методе равномерного поиска. Делить 5 точками и несколько раз или сразу разделить на 25.

# x [0,1]
# требуемая точность
eps = 0.1
n = 24 # если меньше совсем плохо работает, количество разбиений внутри отрезка
# границы интервала
a = 0 + 1e-12  # надо отступ
b = 1


def F(x):
    return 2 * x + 1 / x


# метод равномерного поиска
def UniSearch():
    a_all = []
    b_all = []
    a_new = a
    b_new = b
    print("Метод равномерного поиска")
    count = 0
    X_all = []
    F_all = []
    a_all.append(a_new)
    b_all.append(b_new)
    while (b_new - a_new > eps):
        h = (b_new - a_new) / n
        x = [a_new + h * i for i in range(n + 1)]
        X_all.append(x)
        f_x = [F(i) for i in x]
        F_all.append(f_x)
        count += len(f_x)
        j = f_x.index(min(f_x))  # индекс минимального элемента
        # новый интервал неопределённости j-1 j+1 *
        if j == 0:
            a_new = x[j]
        else:
            a_new = x[j - 1]
        if j == len(x) - 1:
            b_new = x[j]
        else:
            b_new = x[j + 1]
        a_all.append(a_new)
        b_all.append(b_new)
    print("Итоговый интервал: [", a_new, ";", b_new, "]")
    print("Количество обращений к функции:", count)
    return count#, a_all, b_all, X_all, F_all

# метод золотого сечения
def GoldenSelectionSearch():
    a_new = a
    b_new = b
    i = 1
    alf = (3 - 5 ** (1 / 2)) / 2
    count = 0
    lam_all = []
    mu_all = []
    while (b_new - a_new) > eps:
        lam_k = a_new + alf * (b_new - a_new)
        mu_k = a_new + b_new - lam_k
        #print("Итерация: ", i, "; lambda_k = ", lam_k, "; mu_k = ", mu_k)
        i += 1
        if F(lam_k) < F(mu_k):
            b_new = mu_k
        else:
            a_new = lam_k
        if count == 0:
            count += 2
        else:
            count += 1
        lam_all.append(lam_k)
        mu_all.append(mu_k)
    print("Итоговый интервал: [", a_new, ";", b_new, "]")
    print("Количество обращений к функции:", count)
    return count#, lam_all, mu_all

'''
рисование графика исхожной функции
x = np.linspace(a, b, num = 100)
y = F(x)
x1 = [a, b]
y1 = [F(0.5 ** 0.5), F(0.5 ** 0.5)]
plt.plot(x,y, label = 'F(x) = 2*x + 1/x')
plt.plot(0.5 ** 0.5,F(0.5 ** 0.5), 'r*', label = 'min')
plt.grid()
plt.legend()
plt.show()'''

''' Одно епс, разные разбиения, метод равномерного поиска
n_all = []
c_all = []
while (n <= 30):
    print(n)
    n_all.append(n)
    c_all.append( UniSearch())
    n += 1

fig, ax = plt.subplots()
ax.plot(n_all, c_all, 'o',
        color='indigo')
plt.grid()
plt.show()
'''

''' метод золотого сечения
GoldenSelectionSearch()
'''

'''графическая иллюстрация метода равномерного поиска
c, a_all, b_all, X_all, F_all = UniSearch()
x = np.linspace(0.1, b, num = 100)
y = F(x)
x1 = [a, b]
y1 = [F(0.5 ** 0.5), F(0.5 ** 0.5)]
plt.plot(x,y, label = 'F(x) = 2*x + 1/x')
plt.plot(0.5 ** 0.5,F(0.5 ** 0.5), 'r*', label = 'min')

i=2
plt.plot([a_all[i] ,a_all[i]  ], [2, F(0.1)], 'k--')
plt.plot([b_all[i], b_all[i]], [2, F(0.1)], 'm--')
if (i == 0):
    F_all[0][0] = F(0.1)
plt.plot(X_all[i], F_all[i], 'g^')
ind = F_all[i].index((min(F_all[i])))
plt.plot(X_all[i][ind], F_all[i][ind], 'r^')

plt.grid()
plt.legend()
plt.show()
'''

'''Графическая интерперетация метода золотого сечения
i=0
count, lam_all, mu_all = GoldenSelectionSearch()
x = np.linspace(0.1, b, num = 100)
y = F(x)
x1 = [a, b]
y1 = [F(0.5 ** 0.5), F(0.5 ** 0.5)]
plt.plot(x, y, label = 'F(x) = 2*x + 1/x')
plt.plot(0.5 ** 0.5, F(0.5 ** 0.5), 'r*', label = 'min')

plt.plot([lam_all[i], lam_all[i]], [2, F(0.1)], 'k--')
plt.plot([mu_all[i], mu_all[i]], [2, F(0.1)], 'm--')

plt.grid()
plt.legend()
plt.show()'''

'''Метод равномерного поиска Исследование количества итераций от точности
for i in range(10):
    print("eps = ",eps)
    count = UniSearch()
    eps = 0.1 * (0.1 ** i)
    print("count = ", count)'''

'''Метод Золотого сечения Исследование количества итераций(обращений) от точности
for i in range(10):
    print("eps = ", eps)
    count = GoldenSelectionSearch()
    eps = 0.1 * (0.1 ** i)
    print("count = ", count)'''