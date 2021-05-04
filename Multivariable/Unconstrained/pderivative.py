from math import *

def partialDerivative(func,X,Y,diff):
    """
    func -> objective function (str)
    X: list of string variables of interest (list)
    Y: point of evaluation (list)
    diff: differentiate with respect to this variable (str)
    """
    D = {}
    D1 = {}
    h = 1e-10
    test = list()
    
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    if diff in X:
        if len(X) == len(Y):
            
            #MAPPING: X[i] -> Y[i]
            for k in range(0,len(X)):
                D[X[k]] = Y[k]

            seg1 = eval(func,D, math_ops) #f(x1,x2,..xi,...,xn)
            D1 = D        
            for i in D1.keys():
                if i == diff:
                    D1[i] = D1[i] + h
                else:
                    pass

            seg2 = eval(func,D1, math_ops) #f(x1,x2,...,(xi+h),...,xn)
            P = (seg2 - seg1)/h
            return P
    
        else:
            raise ValueError("dim(X) is not equal to dim(Y).")
    else:
        raise ValueError(diff + " is not specified in vector X.")
        
def multiOut(func,X,Y):
    """
    task: computes function
    func: function of interest (str)
    X: list of string variables of interest (list)
    Y: point of evaluation (list)
    """
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    D = {}
    
    for k in range(0,len(X)):
        D[X[k]] = Y[k]
    
    
    y = eval(func, D, math_ops)
    return y

def grad(func,X,Y):
    """
    task: computes gradient of F(x1,x2,...,xn) at a point 
    func: objective function (str)
    X: list of string variables of interest (list of strings)
    Y: point of evaluation (list of integers)
    output: list of integers (gradient of F)
    """
    grad_output = []
    P = None
    
    if len(X) == len(Y):
        for k in range(0,len(X)):
            P = partialDerivative(func,X,Y,X[k])
            grad_output.append(P)
    else:
        raise ValueError("dim(X) is not equal to dim(Y).")
        
    return grad_output