from scipy.optimize import linprog
def simplex(obj,lhs_ineq,rhs_ineq,bnd):
    opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
                  bounds=bnd,
                  method="simplex")
    return opt.x