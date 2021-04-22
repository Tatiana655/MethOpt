import numpy as np
from scipy.optimize import minimize


def f(x):
    return (x[0] ** 2 +  x[1] ** 2 + np.sin(2 * x[0] + 3 * x[1]) + 3 * x[0] + 2 * x[1])
    #return max((x[0]+1) ** 2 + 5 * x[1] ** 2, (x[0]-1) ** 2 +5*x[1] ** 2)
def gradf(x):
    return np.array([3 + 2 * x[0] + 2 * np.cos(2 * x[0] + 3 * x[1]),
                      2 + 2 * x[1] + 3 * np.cos(2 * x[0] + 3 * x[1])])

def H(x):
    return
    #return np.array([[2 - 4 * np.sin(2 * x[0] + 3 * x[1]), -6 * np.sin(2 * x[0] + 3 * x[1])],
    #                [-6 * np.sin(2 * x[0] + 3 * x[1]), 2 - 9 * np.sin(2 * x[0] + 3 * x[1])]])


def goldenSearch(f, a, b, eps):
    al2 = (3 - 5 ** (1 / 2)) / 2
    count = 0

    (a_k, b_k) = (min(a, b), max(a, b))
    length = b_k - a_k
    if length <= eps:
        return a_k, b_k

    lambda_k = a_k + al2 * length
    mu_k = b_k - al2 * length
    f_lambda_k = f(lambda_k)
    f_mu_k = f(mu_k)
    count += 2

    while length > eps:
        if f_lambda_k < f_mu_k:
            b_k = mu_k
            mu_k = lambda_k
            f_mu_k = f_lambda_k
            length = b_k - a_k
            lambda_k = a_k + al2 * length
            f_lambda_k = f(lambda_k)
            count += 1
        else:
            a_k = lambda_k
            lambda_k = mu_k
            f_lambda_k = f_mu_k
            length = b_k - a_k
            mu_k = b_k - al2 * length
            f_mu_k = f(mu_k)
            count += 1
    return (a_k + b_k) / 2


def StepCrush(f, x, p, grad):
    alpha = 1
    lambd = 0.5
    delta = 0.5
    while f(x + alpha * p) - f(x) > 1 * alpha * delta * np.dot(grad(x), p):
        alpha *= lambd
    return x + alpha * p


def SearchAlpha(f, x, p, grad=0):
    delt = 0.5
    b = delt
    a = 0
    while f(x + a * p) < f(x + b * p):
        a += delt
        b += delt

    alpha = goldenSearch(lambda y: f(x + y * p), a, b, eps)
    return x + alpha * p


def SecondGrad(f, grad, hesse, x0, eps, alpha):
    normGrad = 3
    x = x0
    vec = []
    vec.append(x)
    count = 0
    while normGrad >= eps:
        count += 1
        H = hesse(x)
        b = -1 * grad(x)
        p = np.linalg.solve(H, b)
        x = alpha(f, x, p, grad)
        normGrad = np.linalg.norm(grad(x), ord=2)
        vec.append(x)
    print("итерации", count)
    print("x_k-1 = ", vec[len(vec)-2])
    x = np.round(np.array(x),4)
    return x

def dih(f,a,b,eps):
    ak= a
    bk= b
    while ( bk - ak > eps):
        x_med = (ak + bk) / 2
        sig = 0.1 * (bk - ak)
        f1 = f(x_med - sig)
        f2 = f(x_med + sig)
        if f1 >= f2:
            ak = x_med - sig
        else:
            bk = x_med + sig
    return (bk + ak) / 2

def fibb(f,a,b,eps):
    F = [1,1]
    n = 1 # общее число вычислений функции
    while F[n] <=((b-a) / eps):
        F.append(F[n-1] + F[n])
        n+=1
    lam = a + F[n-2]/F[n] * (b - a)
    mu = a + F[n-1]/F[n] * (b - a)
    k = 1
    ak = a
    bk = b
    while(1):
        if f(lam) > f(mu):
            ak = lam
            lam = mu
            mu = ak + F[n-k-1]/F[n-k] * ( bk - ak )
            if k == n-2:
                break
        else:
            bk = lam
            mu = lam
            lam = ak + F[n-k-2]/F[n-k] * ( bk - ak )
            if k == n-2:
                break
        k += 1
    return (bk+ak)/2

def find_alf(f, gradf, x, param):
    delt = 0.5
    b = delt
    a = 0
    while f(x + a * gradf(x)) > f(x + b * gradf(x)):
        a += delt
        b += delt
    if param == 1:
        alpha = dih(lambda y: f(x + y * gradf(x)), a, b, eps)
    else:
        alpha = fibb(lambda y: f(x + y * gradf(x)), a, b, eps)
    return alpha

def FirstGrad(f, grad, x0, eps, find_alf,param):
    normGrad = 3
    x = x0
    count = 0
    vec = []
    vec.append(x)
    while normGrad >= eps:
        count += 1
        step = find_alf(f,grad,x,param)
        x = x - step * grad(x)
        #print(x)
        #H_x = H(x)
        #w, v = np.linalg.eigh(H_x)
        #print("Собственные числа", w)
        normGrad = np.linalg.norm(grad(x), ord=2)
        vec.append(x)
    print("итерации", count)
    #print("x_k-1 = ", vec[len(vec) - 2])
    x = np.round(np.array(x), 4)
    return x

if __name__ == "__main__":
    x_1= 5
    x0 = np.array([-1, 0])#5*x_1**2]
    eps = 0.1
    for i in range(3):
        print("eps =", eps)
        print("Fib = ",FirstGrad(f,gradf,x0,eps, find_alf,1))
        print("Dih = ",FirstGrad(f, gradf, x0, eps, find_alf,0))

        eps /=10
    sol = minimize(f, x0, method='Nelder-Mead', tol=10 ** -10).x
    print("Sol =",sol)

    '''x0 = [-1, 0]
    eps = 0.1
    sol = minimize(f, x0, method='Nelder-Mead', tol=10 ** -10).x
    while eps > 10 ** -4:
        print(eps)
        #print("2GradAlpha",round(np.linalg.norm(SecondGrad(f, gradf, H, x0, eps, SearchAlpha)-sol),6))
        #print("2GradAlpha-Sol", SecondGrad(f, gradf, H, x0, eps, SearchAlpha))
        #print("2GradDr",round(np.linalg.norm(SecondGrad(f, gradf, H, x0, eps, StepCrush) - sol),6))
        #print("2GradDr-Sol",SecondGrad(f, gradf, H, x0, eps, StepCrush))
        print("OneStep",round(np.linalg.norm(FirstGrad(f, gradf, x0, eps, 0.126) - sol),6))
        print("OneStep-Sol",FirstGrad(f, gradf, x0, eps, 0.126) )
        eps = eps / 10'''
