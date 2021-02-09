import numpy as np

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


# начальные данные
# MxN #3x5
c =  [0,0, 0, 0, 0, 0,1,1,1]

A = [
    [0.6, 0.3, 0.5, -1, 0, 0,1,0,0],
    [0.1, 0.2, 0.1, 0, 1, 0,0,1,0],
    [0.5, 0.3, 0.4, 0, 0, 1 , 0,0,1]
]

b = [120, 30, 100]
A = np.array(A).transpose()
# найти первый опорный вектор
x = find_first_vec(A,b,9)
while (1):
    N_k_p = [index for index, data in enumerate(x) if x[index] > 0]
    N_k_0 = [index for index, data in enumerate(x) if x[index] == 0]
    # количество в N_k_p == 3 иначе пополнить и определитель не 0 # тут падает пока, надо написать функцию
    # ввести N1 и L1?

    Nk = N_k_p
    Lk = N_k_0
    if len(N_k_p) == 2:#надо ещё индексы куда надо добавить
        Nk.append(0)
        for i in range(len(Lk)):
            if (Lk[i] > Nk[0]) & (Lk[i] > Nk[1]):
                A_M_Nk = np.transpose(np.matrix([np.array(A[Nk[0]]), np.array(A[Nk[1]]), np.array(A[Lk[i]])]))
            if (Lk[i] < Nk[0]) & (Lk[i] < Nk[1]):
                A_M_Nk = np.transpose(np.matrix([np.array(A[Lk[i]]), np.array(A[Nk[0]]), np.array(A[Nk[1]])]))
            if (Lk[i] > Nk[0]) & (Lk[i] < Nk[1]):
                A_M_Nk = np.transpose(np.matrix([ np.array(A[Nk[0]]),np.array(A[Lk[i]]), np.array(A[Nk[1]])]))
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
    print("A:", A)
    A_M_Nk = np.transpose(np.matrix([np.array(A[Nk[0]]), np.array(A[Nk[1]]), np.array(A[Nk[2]])]))  # [Nk]#!!!!!
    c_Nk = np.array([c[Nk[0]], c[Nk[1]], c[Nk[2]]])

    Bk = np.linalg.inv(A_M_Nk)
    yk = c_Nk * Bk

    dk = np.transpose(c) - np.array(yk) * np.transpose(np.matrix(A))
    d_non = np.array([])
    d_non = np.append(d_non, dk[0])
    d_Lk = [d_non[index] for index, data in enumerate(d_non) if d_non[index] != 0]
    if np.ma.amin(d_Lk) >= 0:
        sol = x
        break
    else:
        j_k = find_j_k(Lk, d_Lk) #возвращает самый отрицательный
        u_Nk = np.array([])
        u_Nk =np.append(u_Nk, np.transpose(Bk * np.matrix(np.transpose([A[j_k]]))))
        u_k = np.zeros(len(np.transpose(A)[0]))
        u_k[j_k] = -1
        for i in range(len(N_k_p)):
            u_k[N_k_p[i]] = u_Nk[i]
        #если кто-то из Nk <0 то умирай. Всё плохо. тут такого типа нет

        theta_k = x[N_k_p[0]] / u_k[N_k_p[0]]
        for i in range(len(N_k_p)):
            if (u_k[N_k_p[i]] > 0):
                theta_k = min(x[N_k_p[i]] / u_k[N_k_p[i]], theta_k)

        x = x - theta_k * np.array(u_k)
        print("x:", x)
print("res = ",x)
