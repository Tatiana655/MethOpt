import numpy as np

import math as m
import Simplex
def f(x):#fi
    return (x[0] ** 2 +  x[1] ** 2 + np.sin(2 * x[0] + 3 * x[1]) + 3 * x[0] + 2 * x[1])

def gradf(x):
    return np.array([3 + 2 * x[0] + 2 * np.cos(2 * x[0] + 3 * x[1]),
                      2 + 2 * x[1] + 3 * np.cos(2 * x[0] + 3 * x[1])])

def FrankWolf(x0,eps,alf, lam, param):
    count=0
    x_k = x0
    etta =10
    ar_etta = []
    x_res = []
    x_res.append(x0)
    while etta >= eps:
        count += 1
        alf_0 = alf
        obj = gradf(x_k)
        lhs_ineq = [[-1, 1],  # -x+y<=2
                    [0.5, -1],  # x/2-y<=2
                    [-1, -1],  # -x-y <=5
                    ]
        if not param:
            rhs_ineq = [2, 2, 3]
        else:
            rhs_ineq = [0.91476418 -0.12214627, 2, 3]

        bnd = [(-float("inf"), 0),  # Границы x
               (-float("inf"), 1)]  # Границы y

        y_k =  Simplex.simplex(obj,lhs_ineq,rhs_ineq,bnd)
        s_k = np.array(y_k) - np.array(x_k)

        while (f(x_k + alf_0 * s_k) - f(x_k) > 0.5 * alf_0 * np.dot(gradf(x_k), s_k)):
            alf_0 = alf_0 * lam
        x_k = x_k + alf_0 * s_k
        x_res.append(x_k)
        etta = m.fabs(np.dot(gradf(x_k), s_k))
        ar_etta.append(-etta)
    return np.round(x_k,6), count, ar_etta, x_res


sol = [-0.91476418, -0.12214627]

eps = 0.1
alf_0=0.5
lam =0.5
x0 = [-5/6,0]
#не на границе
print("x0=", x0)
print("искомое решение внутри области")
for i in range(4):
    ret = FrankWolf(x0,eps,alf_0,lam,False)
    print("eps = ",eps," x, iter = ",ret[0],ret[1])
    print("x-x*=", round(np.linalg.norm(ret[0]-sol),6))
    eps = round(eps/10,10)

eps = 0.1
print("искомое решение на ребре")
for i in range(4):
    ret = FrankWolf(x0,eps,alf_0,lam,True)
    print("eps = ",eps," x, iter = ",ret[0],ret[1])
    print("x-x*=", round(np.linalg.norm(ret[0] - sol),6))
    eps = round(eps/10,10)

eps =0.5
#доп исследование, проверка леммы
print("Эмпирическая проверка леммы")
print("Точка внутри области")
print("eps =",eps)
res = FrankWolf(x0,eps,alf_0,lam,False)
print("etta[] = ",np.round(res[2],5))
print("x_k[] = ",np.round(res[3],5))
print("f(x_k)[] = ",[np.round((f(i)),5) for i in res[3]])
f_ar = [(f(i)-f(sol)) for i in res[3]]
f_ar = [(k)*f_ar[k] for k in range(len(f_ar))]
print("(f_k-f(x*)) * k [] ==", np.round(f_ar,5))

print("Точка на ребре")

print("eps =",eps)
res = FrankWolf(x0,eps,alf_0,lam,True)
print("etta[] = ",np.round(res[2],5))
print("x_k[] = ",np.round(res[3],5))
print("f(x_k)[] = ",[np.round((f(i)),5) for i in res[3]])
f_ar = [(f(i)-f(sol)) for i in res[3]]
f_ar = [k*f_ar[k] for k in range(len(f_ar))]
print("(f_k-f(x*)) * k [] ==", np.round(f_ar,5))


'''for x in res[3]:
    print("start")
    print(np.round(x,5))
    print(round(-x[0]+x[1],5))
    print(round(x[0]/2-x[1],5))
    print(round(-x[0] - x[1],5))
    print(round(x[1],5))
    print(round(x[0],5))'''
