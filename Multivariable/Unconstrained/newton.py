from pderivative import partialDerivative
from pderivative import multiOut
from pderivative import grad
from second_order import secondPartial
from second_order import secondGrad
from second_order import Hessian
from second_order import inverseMatrix
import numpy as np
from math import *
import statistics
from operator import add
from operator import mul
from operator import sub
from constrained_opt import Jacobian

def NewtonMethod(func,X,Y):
    """
    task: use Newton's method of unconstrained optimization to return {x1,x2,..,xN} for optimum value
    input: objective function (func - str)
    X: list of string variables: {x1,x2,...,xN} (string variables in a list)
    Y: point of evaluation (initial guess) (int/floats in a list)
    
    """
    
    solutionC = []
    H = Hessian(func,X,Y)
    Hinv = inverseMatrix(H)
    guess = Y
    
    error = []
    mean_error = 1
    
    while mean_error > 1e-12:
        gradF = grad(func,X,guess)
        H = Hessian(func,X,guess)
        Hinv = inverseMatrix(H)
        
        mult = np.matmul(Hinv,gradF).tolist()
        solutionC = list(map(sub,guess,mult))
        
        del error[:]
        for k in range(0,len(solutionC)):
            e = abs((solutionC[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solutionC
        mean_error = statistics.mean(error)
        
    if abs((solutionC[0]-guess[0])/guess[0]) > 1000:
        raise ValueError("The guess gives divergent results.")
    
    else:
        pass
    
    return {'optimum solution: ': solutionC, 'error: ': mean_error}

def Marquardt(func,X,Y):
    """
    apply Marquardt algorithm to determine optimum value of f = f(X1,X2,...,XN)
    func - objective function (str)
    X - list of string variables (list) {x1,x2,..,xN}
    Y - initial guess (list)
    
    The Marquardt algorithm is a combination of gradient descent and Newton's method.
    This is version 1 of the Marquardt algorithm; this algorithm version proposes that:
    X(i+1) = X(i) - Hinv*grad(F).transpose
        if f[x(k+1)] < f[x(k)]: gamma(k+1) = gamma(k)/2
        else: gamma(k+1) = 2*gamma(k)
    
    Version 2 will utilize the following development:
    H_bar = H + gamma*I; I = identity matrix, H = Hessian matrix, H_bar = approximate Hessian
    at first: utilize large values of gamma; H_bar = gamma*I; H_bar.inv = (1/gamma)*I
    as you get closer to optimum, use Newton's method definition; H_bar = H
    """
    gamma = 10
    solutionC = []
    mean_error = 1
    guess = Y
    error = []
    
    while mean_error > 1e-12:
        H = Hessian(func,X,guess)
        Hinv = inverseMatrix(H)
        gradF = grad(func,X,guess)
        
        HinvdF = np.matmul(Hinv,gradF).tolist()
        solutionC = list(map(sub,guess,HinvdF))
        
        f1 = multiOut(func,X,guess)
        f2 = multiOut(func,X,solutionC)
        
        if f2 < f1:
            gamma = gamma/2 #f[x(k+1)] < f[x(k)]; gamma(k+1) = gamma(k)/2
        elif f2 > f1:
            gamma = 2*gamma #f[x(k+1)] > f[x(k)]; gamma(k+1) = 2*gamma(k)
        
        
        del error[:]
        for k in range(0,len(solutionC)):
            e = abs((solutionC[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solutionC
        mean_error = statistics.mean(error)
        
    return solutionC

def Marquardtv2(func,X,Y):
    """
    Version 2 of the Marquardt algorithm will utilize the following development:
    H_bar = H + gamma*I; I = identity matrix, H = Hessian matrix, H_bar = approximate Hessian
    at first: utilize large values of gamma; H_bar = gamma*I; H_bar.inv = (1/gamma)*I
    as you get closer to optimum, use Newton's method definition; H_bar = H
    
    apply Marquardt algorithm to determine optimum value of f = f(X1,X2,...,XN)
    func - objective function (str)
    X - list of string variables (list) {x1,x2,..,xN}
    Y - initial guess (list)
    """
    gamma = 5
    mean_error = 1
    guess = Y
    
    
    I = np.identity(len(X)).tolist() #identity matrix
    I = [[int(x) for x in lst] for lst in I]
    
    error = [0,0]
    I_new = []
    
    for k in I:
        I_new.append(list(map((1/gamma).__mul__,k)))
    
    while mean_error > 0.3:
        gradF = grad(func,X,guess)
        comp = np.matmul(I_new,gradF).tolist()
        
        solution = list(map(sub,guess,comp))
        
        del error[:]
        for k in range(0,len(solution)):
            e = abs((solution[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solution
        mean_error = statistics.mean(error)
    
    solutionC = []
    
    while mean_error > 1e-12:
        H = Hessian(func,X,guess)
        Hinv = inverseMatrix(H)
        gradF = grad(func,X,guess)
        
        gradH = np.matmul(Hinv,gradF).tolist()
        
        solutionC = list(map(sub,guess,gradH))
        
        del error[:]
        for k in range(0,len(solutionC)):
            e = abs((solution[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solutionC
        mean_error = statistics.mean(error)
        
    
    return solutionC

def mag(x): 
    """
    computes magnitude of vector [x1,x2,...,xN]
    """
    return sqrt(sum(i**2 for i in x))

def GlobalNewton(func,X,Y):
    """
    utilize the Quasi-Newton algorithm to find optimum coordinates
    steps:
    1. choose a parameter c; c > 0
    2. choose some initial guess Xo (chosen by user)
    3. for k = {0,1,2,....,N}:
    A) H*d(k) = -grad(f(x_k))
        d(k) = -H.inv*grad(f(x_k))
    B) if:
            - (dot(grad(f(x(k))),d(k)))/(mag(f(x(k)))*d(k)) < c
            set: d(k) = -grad(f(x(k)))
    
    C) set p_k = 1/2 (p_k > 0)
    D) x(k+1) = x(k) + p(k)*d(k)
    go back to (A) until error criteria is ultimately met
    
    """
    c = 1
    guess = Y
    pk = 0.05
    error = []
    mean_error = 1
    
    while mean_error > 1e-12:
        H = Hessian(func,X,guess)
        Hinv = inverseMatrix(H)
        H_new = []
        
        for k in Hinv:
            H_new.append([x*-1 for x in k])
        
        gradF = grad(func,X,guess)
        
        dk = np.matmul(H_new,gradF).tolist()
        
        condition = -1*((np.dot(gradF,dk))/(mag(gradF)*mag(dk)))
        
        if condition < c:
            pkdk = list(map((-pk).__mul__,gradF))
        else:
            pkdk = list(map((pk).__mul__,dk))
        
        solution = list(map(add,guess,pkdk))
        
        del error[:]
        for k in range(0,len(solution)):
            e = abs((solution[k]-guess[k])/guess[k])
            error.append(e)
        
        mean_error = statistics.mean(error)
        guess = solution
    
    
    return solution
