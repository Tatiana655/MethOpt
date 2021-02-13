import numpy as np
import matplotlib.pyplot as plt
def F(c, x):
    return sum(c * x)


def find_first_vec(A, b, N):
    A = np.array(A)
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):  # составить матрицу из столбцов, решить методом наусса проверить на положительность
                A_help = np.array([A[i], A[j], A[k]]).transpose()
                if np.linalg.det(A_help) != 0:
                    first_vec = np.linalg.solve(A_help, b)
                    if (np.ma.amin(first_vec) != 0):
                        # заполнить нулями
                        vec = np.zeros(N)
                        vec[i] = first_vec[0]
                        vec[j] = first_vec[1]
                        vec[k] = first_vec[2]
                        print(vec)
                        return vec


def find_j_k(N_k_0, d_k):
    mini = 1
    res = -1
    for i in range(len(N_k_0)):
        if d_k[i] < 0:
            if mini != min(mini, d_k[i]):
                mini = min(mini, d_k[i])
                res = N_k_0[i]
    return res


# Исходная матрица, вектор свободных членов, вектор целевой функции, перый опорный вектор
def SimplexAlg(A, b, c, N, x):
    #print("Начальный вектор: ", x)
    while (1):
        N_k_p = [index for index, data in enumerate(x) if x[index] > 1e-10]
        N_k_0 = [index for index, data in enumerate(x) if x[index] <= 1e-10]
        # количество в N_k_p == 3 иначе пополнить и определитель не 0 # тут падает пока, надо написать функцию
        # ввести N1 и L1?

        Nk = N_k_p
        Lk = N_k_0
        if len(N_k_p) == 2:  # надо ещё индексы куда надо добавить
            Nk.append(0)
            for i in range(len(Lk)):
                if (Lk[i] > Nk[0]) & (Lk[i] > Nk[1]):
                    A_M_Nk = np.transpose(np.matrix([np.array(A[Nk[0]]), np.array(A[Nk[1]]), np.array(A[Lk[i]])]))
                if (Lk[i] < Nk[0]) & (Lk[i] < Nk[1]):
                    A_M_Nk = np.transpose(np.matrix([np.array(A[Lk[i]]), np.array(A[Nk[0]]), np.array(A[Nk[1]])]))
                if (Lk[i] > Nk[0]) & (Lk[i] < Nk[1]):
                    A_M_Nk = np.transpose(np.matrix([np.array(A[Nk[0]]), np.array(A[Lk[i]]), np.array(A[Nk[1]])]))
                if np.linalg.det(A_M_Nk) != 0:
                    if (Lk[i] > Nk[0]) & (Lk[i] > Nk[1]):
                        Nk[2] = Lk[i]
                    if (Lk[i] < Nk[0]) & (Lk[i] < Nk[1]):
                        Nk[2] = Nk[1]
                        Nk[1] = Nk[0]
                        Nk[0] = Lk[i]
                    if (Lk[i] > Nk[0]) & (Lk[i] < Nk[1]):
                        Nk[2] = Nk[1]
                        Nk[1] = Lk[i]
                    Lk.remove(Lk[i])
                    break
        # print("A:", A)
        A_M_Nk = np.transpose(np.matrix([np.array(A[Nk[0]]), np.array(A[Nk[1]]), np.array(A[Nk[2]])]))  # [Nk]#!!!!!
        c_Nk = np.array([c[Nk[0]], c[Nk[1]], c[Nk[2]]])

        Bk = np.linalg.inv(A_M_Nk)
        yk = c_Nk * Bk

        dk = np.transpose(c) - np.array(yk) * np.transpose(np.matrix(A))
        d_non = np.array([])
        d_non = np.append(d_non, dk[0])
        #d_Lk = [d_non[index] for index, data in enumerate(d_non) if abs(d_non[index]) > 1e-10]
        d_Lk = np.zeros(len(Lk))
        for i in range(len(Lk)):
            d_Lk[i] = d_non[Lk[i]]
        if np.ma.amin(d_Lk) >= 0:
            sol = x
            break
        else:
            j_k = find_j_k(Lk, d_Lk)  # возвращает самый отрицательный
            u_Nk = np.array([])
            u_Nk = np.append(u_Nk, np.transpose(Bk * np.matrix(np.transpose([A[j_k]]))))
            u_k = np.zeros(len(np.transpose(A)[0]))
            u_k[j_k] = -1
            count = 0
            for i in range(len(Nk)):
                u_k[Nk[i]] = u_Nk[i]
                if u_k[Nk[i]] <= 0:
                    count += 1
            # если весь u[Nk] <= 0 то умирай. Всё плохо. тут такого типа нет
            if count == len(Nk):
                print("Область неограничена")
                return x
            theta_k = 1000000000
            for i in range(len(Nk)):
                if u_k[Nk[i]] > 0:
                    theta_k = min(x[Nk[i]] / u_k[Nk[i]], theta_k)
            if (theta_k == 1000000000) or (theta_k == 0):
                print("theta <=0:(")
                return x
            x = x - theta_k * np.array(u_k)

            #print("опорный вектор: ", x)
    return x


# начальные данные
# MxN #3x5
#ff

c = [0, 0, 0, 0, 0, 0, 1, 1, 1]

A = [
    [0.6, 0.3, 0.5, -1, 0, 0, 1, 0, 0],
    [0.1, 0.2, 0.1, 0, 1, 0, 0, 1, 0],
    [0.5, 0.3, 0.4, 0, 0, 1, 0, 0, 1]
]

b = [120, 30, 100]
A = np.array(A).transpose()
# найти первый опорный вектор
x = [0, 0, 0, 0, 0, 0, 120, 30, 100]  # find_first_vec(A, b, 9)
x = SimplexAlg(A, b, c, 9, x)
print("Result: ", x)

c = [-7, -8.2, -8.6, 0, 0, 0]

A = [
    [0.6, 0.3, 0.5, -1, 0, 0],
    [0.1, 0.2, 0.1, 0, 1, 0],
    [0.5, 0.3, 0.4, 0, 0, 1]
]

b = [120, 30, 100]
A = np.array(A).transpose()
x1 = np.delete(x, [6, 7, 8])
x1 = SimplexAlg(A, b, c, 6, x1)
print("Finish: ", x1)
print("F(x) = ", F(c, x1))
solution = x1
solutionF = F(c, x1)

# Двойственная задача:
# начальные данные
# MxN #3x5
c = [0, 0, 0, 0, 0, 0, 1, 1, 1]

A = [
    [-0.6, 0.1, 0.5, -1, 0, 0, 1, 0, 0],
    [-0.3, 0.2, 0.3, 0, -1, 0, 0, 1, 0],
    [-0.5, 0.1, 0.4, 0, 0, -1, 0, 0, 1]
]

b = [7, 8.2, 8.6]

A = np.array(A).transpose()
# найти первый опорный вектор
x = [0, 0, 0, 0, 0, 0, 7, 8.2, 8.6]  # find_first_vec(A, b, 9)
x = SimplexAlg(A, b, c, 9, x)
print("Result: ", x)

c = [-120, 30, 100, 0, 0, 0]

A = [
    [-0.6, 0.1, 0.5, -1, 0, 0],
    [-0.3, 0.2, 0.3, 0, -1, 0],
    [-0.5, 0.1, 0.4, 0, 0, -1]
]

b = [7, 8.2, 8.6]
A = np.array(A).transpose()
x1 = np.delete(x, [6, 7, 8])
x1 = SimplexAlg(A, b, c, 6, x1)
print("Finish: ", x1)
print("F(x) = ", F(c, x1))
print("Вносим изменения в b ")
print_F = np.zeros((3,10))
print_delt = np.zeros((3,10))
for k in range(3):
    for i in 0,1,2,3,4,5,6,7,8,9:
        c_ch = [0, 0, 0, 0, 0, 0, 1, 1, 1]

        A_ch = [
            [0.6, 0.3, 0.5, -1, 0, 0, 1, 0, 0],
            [0.1, 0.2, 0.1, 0, 1, 0, 0, 1, 0],
            [0.5, 0.3, 0.4, 0, 0, 1, 0, 0, 1]
        ]
        delt = 10**(-i)
        print_delt[k][i] = delt
        print(delt)
        if (k==0):
            b_ch = [120+delt, 30, 100]
        if (k==1):
            b_ch = [120, 30 + delt, 100]
        if (k == 2):
            b_ch = [120, 30, 100+delt]
        A_ch = np.array(A_ch).transpose()
        # найти первый опорный вектор
        x_ch = [0, 0, 0, 0, 0, 0, b_ch[0], b_ch[1], b_ch[2]]  # find_first_vec(A, b, 9)
        x_ch = SimplexAlg(A_ch, b_ch, c_ch, 9, x_ch)
        #print("Result: ", x_ch)

        c_ch = [-7, -8.2, -8.6, 0, 0, 0]

        A_ch = [
            [0.6, 0.3, 0.5, -1, 0, 0],
            [0.1, 0.2, 0.1, 0, 1, 0],
            [0.5, 0.3, 0.4, 0, 0, 1]
        ]

        #b_ch = [120+1**(-i), 30, 100]
        A_ch = np.array(A_ch).transpose()
        x1_ch = np.delete(x_ch, [6, 7, 8])
        x1_ch = SimplexAlg(A_ch, b_ch, c_ch, 6, x1_ch)

        print_F[k][i] = np.sum(np.abs(x1_ch - solution)**2)**(1./2)
        print("||x - x_ch|| = ", np.sum(np.abs(x1_ch - solution)**2)**(1./2))
        print("|F(x) - F_ch(x)| = ", abs(solutionF - F(c_ch, x1_ch)))
#print(np.finfo(np.float64))
line1,  = plt.plot(print_delt[0], print_F[0], 'b.--', linewidth=1.0)

line2,  =plt.plot(print_delt[1], print_F[1], 'g.--', linewidth=1.0)

line3,  =plt.plot(print_delt[2], print_F[2], 'r.--', linewidth=1.0)

plt.legend( (line1, line2, line3), ('b_1+delt', 'b_2+delt', 'b_3+delt'))


plt.ylabel('||x-x_changed||')
plt.xlabel('Delta_b')
plt.grid()
plt.show()