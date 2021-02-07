import numpy as np


def find_first_vec(A, b, N):
    A = np.array(A).transpose()
    for i in range(N):
        for j in range(i, N):
            for k in range(j, N):  # составить матрицу из столбцов, решить методом наусса проверить на положительность
                A_help = np.array([A[i], A[j], A[k]]).transpose()
                if np.linalg.det(A_help) != 0:
                    first_vec = np.linalg.solve(A_help, b)
                    if (first_vec[0] != first_vec[1]) & (first_vec[0] != first_vec[2]) & (
                            first_vec[2] != first_vec[1]) & (np.ma.amin(first_vec) >= 0):
                        # заполнить нулями
                        vec = np.array([0., 0., 0., 0., 0., 0.])
                        vec[i] = first_vec[0]
                        vec[j] = first_vec[1]
                        vec[k] = first_vec[2]
                        return vec


def find_j_k(N_k_0, d_k):
    for i in range(len(N_k_0)):
        if (d_k[N_k_0[i]] < 0):
            return N_k_0[i]
    return -1


# начальные данные
# MxN #3x5
M = 3
N = 5
c = [-7, -8.2, -8.6, 0, 0, 0]

A = [
    [0.6, 0.3, 0.5, -1, 0, 0],
    [0.1, 0.2, 0.1, 0, 1, 0],
    [0.5, 0.3, 0.4, 0, 0, 1]
]

b = np.array([120, 30, 100]).transpose()

# транспонировали, чтобы добыть столбцы
# найти первый опорный вектор
x = find_first_vec(A, b, N)
while (1):
    N_k_p = [index for index, data in enumerate(x) if x[index] != 0]
    N_k_0 = [index for index, data in enumerate(x) if x[index] == 0]

    A = np.array(A).transpose()
    # if на случай если вдруг меньше, определитель не 0

    A_help = np.matrix([np.array(A[N_k_p[0]]), np.array(A[N_k_p[1]]), np.array(A[N_k_p[2]])])  # [Nk]#!!!!!

    L_k = N_k_0  # N-N_k
    c_help = np.array([c[N_k_p[0]], c[N_k_p[1]], c[N_k_p[2]]])  # [Nk]

    B = np.linalg.inv(A_help)
    y_k = np.transpose(c_help) * B

    d_k = np.transpose(c) - np.array(y_k) * np.transpose(np.matrix(A))
    d_k = np.array(d_k)[0]

    d_k_help = np.array([d_k[N_k_0[0]], d_k[N_k_0[1]], d_k[N_k_0[2]]])

    if (np.ma.amin(d_k_help) >= 0):
        break
    j_k = find_j_k(N_k_0, d_k)

    # print(A[j_k])
    # print(B)
    u_k_help = (B * np.matrix(np.transpose([A[j_k]])))

    u_k = [0, 0, 0, 0, 0, 0]
    u_k[j_k] = -1
    u_k[N_k_p[0]] = u_k_help[0, 0]
    u_k[N_k_p[1]] = u_k_help[1, 0]
    u_k[N_k_p[2]] = u_k_help[2, 0]
    theta_k = x[N_k_p[0]] / u_k[N_k_p[0]]
    for i in range(len(N_k_p)):
        if (u_k[N_k_p[i]] > 0):
            theta_k = min(x[N_k_p[i]] / u_k[N_k_p[i]], theta_k)
    x_theta = x - theta_k * np.array(u_k)
    print(x_theta)
    print("F = ", np.array(c) * np.transpose(x_theta))

print(c*x_theta)