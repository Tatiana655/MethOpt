import numpy as np


def find_j_k(N_k_0, d_k):
    for i in range(len(N_k_0)):
        if (d_k[N_k_0[i]] < 0):
            return N_k_0[i]
    return -1


# начальные данные
# MxN #3x5
c = [-8.2, -9, -9.6, 1, -1, -1, 0, 0, 0]  # [-7, -8.2, -8.6, 0, 0, 0]

A = [
    [0.6, 0.3, 0.5, -1, 0, 0, 1, 0, 0],
    [0.1, 0.2, 0.1, 0, 1, 0, 0, 1, 0],
    [0.5, 0.3, 0.4, 0, 0, 1, 0, 0, 1]
]

b = [120, 30, 100]

# найти первый опорный вектор
x = np.array([0, 0, 0, 0, 0, 0, 120, 30, 100])
while (1):
    N_k_p = [index for index, data in enumerate(x) if x[index] >= 0]
    N_k_0 = [index for index, data in enumerate(x) if x[index] == 0]
    # количество в _k_p == 3 иначе пополнить и определитель не 0
    # ввести N1 и L1?

    Nk = N_k_p
    Lk = N_k_0

    A = np.array(A).transpose()
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
        j_k = find_j_k(Lk, d_Lk) #возвращает номер первого отрицательного
        u_Nk = np.array([])
        u_Nk =np.append(u_Nk, np.transpose(Bk * np.matrix(np.transpose([A[j_k]]))))
        u_k = np.zeros(9)
        u_k[j_k] = -1
        for i in range(len(N_k_p)):
            u_k[N_k_p[i]] = u_Nk[i]
        #если кто-то из Nk <0 то умирай. Всё плохо. тут такого типа нет

        theta_k = x[N_k_p[0]] / u_k[N_k_p[0]]
        for i in range(len(N_k_p)):
            if (u_k[N_k_p[i]] > 0):
                theta_k = min(x[N_k_p[i]] / u_k[N_k_p[i]], theta_k)

        x = x - theta_k * np.array(u_k)
        print(x)
print(x)