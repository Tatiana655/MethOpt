import numpy as np
from scipy.optimize import linprog
import math as m

def f(x):#fi
    return (x[0] ** 2 +  x[1] ** 2 + np.sin(2 * x[0] + 3 * x[1]) + 3 * x[0] + 2 * x[1])

def gradf(x):
    return np.array([3 + 2 * x[0] + 2 * np.cos(2 * x[0] + 3 * x[1]),
                      2 + 2 * x[1] + 3 * np.cos(2 * x[0] + 3 * x[1])])


#[-0.91476418; -0.12214627]
#x<0 y<1; y-x=2,y+x=+2, y-x/2=-2
eps = 0.001
alf_0=0.5
lam =0.5
x0 = [-0.2,0.6]
#print(gradf(x0))
#решить симлекс методом min gradf x
x_k = x0
etta = 10
while  etta >= eps:
    alf_0 = 0.5
    obj = gradf(x_k)
    lhs_ineq = [[-1,   1], #-x+y<=2
                [0.5, -1], #x/2-y<=2
                [-1,  -1], #-x-y <=5
    ]

    rhs_ineq = [2,2,1.6]

    bnd = [(-float("inf"), 0),  # Границы x
           (-float("inf"),1)]  # Границы y

    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                  bounds=bnd,
                  method="simplex")

    y_k = opt.x
    s_k = np.array(y_k) - np.array(x_k)


    while (f(x_k + alf_0*s_k)-f(x_k) > 0.5 * alf_0 * np.dot(gradf(x_k), s_k)):
        alf_0 = alf_0 * lam
    x_k = x_k + alf_0 * s_k
    etta =  m.fabs(np.dot(gradf(x_k), s_k))
print(x_k)

