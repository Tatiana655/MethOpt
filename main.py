import numpy as np

# типа найти максимальную разность и вернуть индексы (не запутаться бы в них). тут тоже питоновские штуки
def rightPoint(C, u,v):
    max1 = -1
    max_i = -1
    max_j = -1
    for j in range(len(v)):
        for i in range(len(u)):
            if v[j] - u[i] > C[i][j]:
              if (max1 != max(max1, abs(v[j] - u[i] - C[i][j]))):
                max_i = i#string u
                max_j = j#col v
                max1 = max(max1, abs(v[j] - u[i] - C[i][j]))
    return max_i, max_j




C = [
    [5, 3, 8, 2, 5],
    [6, 2, 5, 1, 1],
    [4, 4, 7, 1, 3],
    [7, 6, 8, 9, 4]
]
a = [16, 13, 8, 7, 9]

b = [8, 20, 6, 9]

#первый опорный вектор, полученный методом С-З угла
x1 = [
    [8,  0, 0, 0, 0],
    [8, 12, 0, 0, 0],
    [0,  1, 5, 0, 0],
    [0,  0, 3, 7, 5]
]

A = np.zeros((len(C), len(C[0])))
for i in range(len(C)):
    for j in range(len(C[0])):
        if (x1[i][j] != 0):
            A[i][j] = C[i][j]
print(A)
#число заполненных клеток n+m-1 (верно)

v = np.zeros(len(x1[0]))
u = np.zeros(len(x1))
v[0] = 8
# ут какие-то питоновские фишки можно попробовать
for i in range(len(C)-1):# строки
    for j in range(len(C[0])):# столбцы
        if x1[i][j] != 0:
            v[j] = x1[i][j] + u[i]
            u[i+1] = -x1[i+1][j] + v[j]

for j in range(len(C[0])):
    if x1[len(u)-1][j] !=0 :
        v[j] = x1[len(u)-1][j] + u[len(u)-1]

print(u, v)

#проверяем оптимальность в пустых клетках типа перебором или по хитрому
# Beda if v_j - u_i > cij

i, j = rightPoint(C, u, v)
if i == -1:
    print("this is opt vec")
    exit()
print(i, j)
#цикл пересчёта


