import math
from operator import add
from operator import mul
from operator import sub
import statistics
from pderivative import partialDerivative
from pderivative import multiOut
from pderivative import grad

def multiDescent(func,X,Y):
    """
    task: use gradient descent to find optimal points of F = F(x1,x2,..,xn) (MINIMA)
    func: objective function (str)
    X: list of variables (strings)
    Y: initial guess for optimal value (list of integers/floats)
    """
    guess = Y
    gamma = 0.0001
    solution = list()
    error = list()
    mean_error = 1
    
    
    while mean_error > 1e-12:
        gradF = grad(func,X,guess)
        solution = list(map(sub,guess,list(map((gamma).__mul__, gradF))))
        
        del error[:]
        for k in range(0,len(solution)):
            e = abs((solution[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solution
        mean_error = statistics.mean(error)
    
    if abs((solution[0]-guess[0])/guess[0]) > 1000:
        raise ValueError("The guess gives divergent results, or there is no minima.")
    
    else:
        pass
    
    return {'optimum solution: ': solution, 'error: ': mean_error}

def multiAscent(func,X,Y):
    """
    task: use gradient ascent to find optimal points of F = F(x1,x2,..,xn) (MAXIMA)
    func: objective function (str)
    X: list of variables (strings)
    Y: initial guess for optimal value (list of integers/floats)
    """
    guess = Y
    gamma = 0.0001
    solution = list()
    error = list()
    mean_error = 1
    
    
    while mean_error > 1e-12:
        gradF = grad(func,X,guess)
        solution = list(map(add,guess,list(map((gamma).__mul__, gradF))))
        
        del error[:]
        for k in range(0,len(solution)):
            e = abs((solution[k]-guess[k])/guess[k])
            error.append(e)
        
        guess = solution
        mean_error = statistics.mean(error)
    
    if abs((solution[0]-guess[0])/guess[0]) > 1000:
        raise ValueError("The guess gives divergent results, or there is no maxima.")
    
    else:
        pass
    
    return {'optimum solution: ': solution, 'error: ': mean_error}

def multiGrad(func,X,Y,option = None):
    """
    task: calculate the optimum point for a multivariable function: F = F(x1,x2,...,xn)
    func: objective function (str)
    X: list of variables (strings) {x1,x2,...,xn}
    Y: initial guess for the optimum value (list of int/floats)
    option: choose whether you want to solve maxima or minima. default option: minima
    
    output: {solution, mean error} (dict)
    """
    
    solution = []
    
    if (option == None) or (option.lower() == 'minima'):
        solution = multiDescent(func,X,Y)
    elif (option.lower() == 'maxima'):
        solution = multiAscent(func,X,Y)
    
    return solution