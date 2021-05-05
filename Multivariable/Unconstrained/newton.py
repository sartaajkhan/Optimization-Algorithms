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