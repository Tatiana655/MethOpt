import numpy as np
from scipy.optimize import minimize


def f(x):
    #return (x[0] ** 2 +  x[1] ** 2 + np.sin(2 * x[0] + 3 * x[1]) + 3 * x[0] + 2 * x[1])
    return max((x[0]+1) ** 2 + 5 * x[1] ** 2, (x[0]-1) ** 2 +5*x[1] ** 2)
def gradf(x):
    f1 = (x[0]+1) ** 2 + 5 * x[1] ** 2
    f2 = (x[0]-1) ** 2 + 5 * x[1] ** 2
    if f1>f2:
        return np.array([2 * (x[0] + 1), 10 * x[1]])
    else:
        return np.array([2 * (x[0] - 1), 10 * x[1]])
    #return np.array([3 + 2 * x[0] + 2 * np.cos(2 * x[0] + 3 * x[1]),
    #                  2 + 2 * x[1] + 3 * np.cos(2 * x[0] + 3 * x[1])])

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

def FirstGrad(f, grad, x0, eps, alpha):

    normGrad = 3
    x = x0
    count = 0
    vec = []
    vec.append(x)
    while normGrad >= eps:
        count += 1
        x = x - alpha * grad(x)
        print(x)
        #H_x = H(x)
        #w, v = np.linalg.eigh(H_x)
        #print("Собственные числа", w)
        normGrad = np.linalg.norm(grad(x), ord=2)
        vec.append(x)
    print("итерации", count)
    print("x_k-1 = ", vec[len(vec) - 2])
    x = np.round(np.array(x), 4)
    return x

if __name__ == "__main__":
    x0 = [0, 0]
    eps = 0.1
    print(FirstGrad(f, gradf, x0, eps, 0.5))



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
