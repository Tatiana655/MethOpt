import numpy as np


# типа найти максимальную разность и вернуть индексы (не запутаться бы в них). тут тоже питоновские штуки
def rightPoint(C, u, v):
    max1 = -1
    max_i = -1
    max_j = -1
    cmax = -1
    for j in range(len(v)):
        for i in range(len(u)):
            if (v[j] - u[i]) > C[i][j]:
                if max1 < max(max1, abs((v[j] - u[i]) - C[i][j])):
                    max_i = i  # string u
                    max_j = j  # col v
                    max1 = max(max1, abs((v[j] - u[i]) - C[i][j]))
                    cmax = C[i][j]
                if max1 == abs((v[j] - u[i]) - C[i][j]) and cmax < C[i][j]:
                    max_i = i  # string u
                    max_j = j  # col v
                    max1 = max(max1, abs((v[j] - u[i]) - C[i][j]))
                    cmax = C[i][j]
    return max_i, max_j


array = []  # не  забыть добавить сюда первый элемент in code


def Modi(bol, A, i, j, flag):
    elem = []
    if (len(array) == 2) and (array[0][0] == array[1][0] and array[0][1] == array[1][1] and flag > 0):
        return 0
    if i == array[0][0] and j == array[0][1] and flag > 0:
        return 1

    if bol:  # ищем элемент на вертикали
        for i_k in range(len(A)):  # or единственый нулевой элемент тот что в самом начале подошёл
            if ((i_k != i) and (A[i_k][j] != 0) and (array.count([i_k, j]) == 0)) or (
                    (i_k == array[0][0]) and (j == array[0][1])):
                elem.append([i_k, j])
                # array.append([i_k, j])
    else:  # ищем элемент на горизонтали
        for j_k in range(len(A[0])):
            if ((j_k != j) and (A[i][j_k] != 0) and (array.count([i, j_k]) == 0)) or (
                    (i == array[0][0]) and (j_k == array[0][1])):
                elem.append([i, j_k])
                # array.append([i, j_k])
    # наверное тут надо получать все элеемнты и бегать циклом и вызывать моди
    # и где-то надо чекнуть, что элементы не повторяются,в if'е, наверное
    if not elem:
        return 0
    for e in elem:
        array.append(e)
        if Modi(not bol, A, e[0], e[1], 1) == 1:
            return 1
        else:
            array.remove(e)


def findMin(array, x0):
    mina = 10e5
    for k in range(0, len(array), 2):
        mina = min(mina, x0[array[k][0]][array[k][1]])
    return mina

def F(x1):
    s = 0
    for i in range(len(C)):
        for j in range(len(C[0])):
            s += C[i][j] * x1[i][j]
    return  s

def Alg(x1):
    while (1):
        # A = np.zeros((len(C), len(C[0])))
        # for i in range(len(C)):
        #    for j in range(len(C[0])):
        #        if (x1[i][j] != 0):
        #            A[i][j] = C[i][j]
        # print(A)

        # число заполненных клеток n+m-1 (верно)

        #        v = np.zeros(len(x1[0]))
        #        u = np.zeros(len(x1))
        # v[0] = x1[0][0] #hardcode!!!
        # тут какие-то питоновские фишки можно попробовать
        # считаем u и v тут неверно, надо слау составить
        #        for i in range(len(C) - 1):  # строки
        #            for j in range(len(C[0])):  # столбцы
        #                if x1[i][j] != 0:
        #                    v[j] = C[i][j] + u[i]
        #                    u[i + 1] = -C[i + 1][j] + v[j]

        #        for j in range(len(C[0])):
        #            if x1[len(u) - 1][j] != 0:
        #                v[j] = C[len(u) - 1][j] + u[len(u) - 1]

        # matrix = np.zeros((len(array) + 1, len(array) + 1))
        # b = np.zeros(len(array) + 1)
        # for i in range(len(array)):
        #      matrix[i][array[i][0]] = -1
        #     matrix[i][len(x1)+array[i][1]] = 1
        #      b[i] = C[array[i][0]][array[i][1]]
        #   matrix[len(array)][0] = -1
        # print("u,v : ", u, v)
        #  vec = np.linalg.solve(matrix, b)
        #  u = vec[0:len(x1)-1]
        # v = vec[len(x1):len(x1)+len(x1[0])-1]
        # проверяем оптимальность в пустых клетках типа перебором или по хитрому
        # Beda if v_j - u_i > cij
        matrix = np.zeros((len(x1)+len(x1[0]), (len(x1)+len(x1[0]))))
        k = 0
        b = np.zeros(len(x1)+len(x1[0]))
        for i in range(len(x1)):#u
            for j in range(len(x1[0])):#v
                if x1[i][j] != 0:
                    matrix[k][i] = -1
                    matrix[k][j+len(x1)] = 1
                    b[k] = C[i][j]
                    k += 1
        matrix[k][0] = -1

        vec = np.linalg.solve(matrix, b)
        u = vec[0:len(x1)]
        v = vec[len(x1): (len(x1)+len(x1[0]))]
        print("u = ", u)
        print("v = ", v)
        i, j = rightPoint(C, u, v)
        if (i == -1):
            print("Оптимальный вектор:")
            return x1
        print("индексы, где нарушается условие", i, j)
        # цикл пересчёта

        array.append([i, j])
        Modi(True, x1, i, j, 0)  # Находит цикл пересчёта, записывает его в array
        print("цикл пересчёта: ", array)
        # там перый и последный элементы повторяются
        array.remove(array[0])
        # print(array)

        # найди минимум из "-" клеток
        mina = findMin(array, x1)
        print("min- = ", mina)
        # пересчёт объёмов груза:
        i = 0
        while i < len(array):  # "-"
            x1[array[i][0]][array[i][1]] -= mina
            i += 2

        i = 1
        while i < len(array):  # "+"
            x1[array[i][0]][array[i][1]] += mina
            i += 2

        print("new x1:", x1)
        print("new F(x1):", F(x1))
        array.clear()
        # снова проверяем оптимальность и пересчёт плана


C = [
    [5, 3, 8, 2, 5],
    [6, 2, 5, 1, 1],
    [4, 4, 7, 1, 3],
    [7, 6, 8, 9, 4]
]
a = [16, 13, 8, 7, 9]

b = [8, 20, 6, 19]

# первый опорный вектор, полученный методом С-З угла
x1 = [
    [8, 0, 0, 0, 0],
    [8, 12, 0, 0, 0],
    [0, 1, 5, 0, 0],
    [0, 0, 3, 7, 9]
]
print("Начальный вектор: ", x1, "\n\n")
x1 = Alg(x1)
print(x1)

print("F_min = ", F(x1))
