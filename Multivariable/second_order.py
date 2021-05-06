from pderivative import partialDerivative
from pderivative import multiOut
from pderivative import grad
from multivar_optimization import multiGrad
from math import *

def secondPartial(func,X,Y,diff1,diff2):
    """
    func -> objective function (str)
    X: list of string variables of interest (list)
    Y: point of evaluation (list)
    diff1: differentiate with respect to this variable (str)
    diff2: differentiate a second time with respect to this variable (str)
    
    output: second partial derivative
    lim h->0 (f(x1,x2,..,(xL+h),..,(xU+h),..,xN) - f(x1,x2,..,(xL+h),..,xU,..xN) - f(x1,..,xL,..,(xU+h),..,xN) + f(
    x1,x2,...,xN)/(h**2)
    """
    
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    h = 0.00001
    D = {}
    D1 = {}
    D2 = {}
    D3 = {}
    
    if len(X) == len(Y):
        if diff1 != diff2:
            #MAPPING: X[i] -> Y[i]:
            for k in range(0,len(X)):
                D[X[k]] = Y[k] #{'x1':A1, 'x2': A2, ...., 'xN':AN}

            seg1 = eval(func,D,math_ops) #f(x1,x2,....,xN)
            
            #f(x1,x2,...,(xL+h),..,(xU+h),...,xN)
            for k in D.keys():
                if (k == diff1) or (k == diff2):
                    D1[k] = D[k] + h
                else:
                    D1[k] = D[k]
            
            seg2 = eval(func,D1,math_ops)
            
            #f(x1,x2,...,(xL+h),...,xU,...,xN)
            for k in D.keys():
                if (k == diff1):
                    D2[k] = D[k] + h
                else:
                    D2[k] = D[k]
            
            seg3 = eval(func, D2, math_ops)
            
            #f(x1,x2,...,xL,...,(xU+h),...,xN)
            for k in D.keys():
                if (k == diff2):
                    D3[k] = D[k] + h
                else:
                    D3[k] = D[k]
            
            seg4 = eval(func,D3,math_ops)
            
            return (seg2 - seg3 - seg4 + seg1)/(h**2)
        
        elif (diff1 == diff2):
            
            for k in range(0,len(X)):
                D[X[k]] = Y[k] #{'x1':A1, 'x2': A2, ...., 'xN':AN}

            seg1 = eval(func,D,math_ops) #f(x1,x2,....,xN)
            
            #f(x1,x2,...,(xL+h),...,xN)
            for k in D.keys():
                if (k == diff1):
                    D1[k] = D[k] + h
                else:
                    D1[k] = D[k]
            
            seg2 = eval(func,D1,math_ops)
            
            #f(x1,x2,...,(xL+2h),...,xU,...,xN)
            for k in D.keys():
                if (k == diff1):
                    D2[k] = D[k] + 2*h
                else:
                    D2[k] = D[k]
            
            seg3 = eval(func, D2, math_ops)
            
            return (seg3 - 2*seg2 + seg1)/(h**2)
    
    else:
        raise ValueError("dim(X) is not equal to dim(Y).")

def secondGrad(func,X,Y,diff1):
    """
    input: objective function, func (str)
    X: list of variables (list of str)
    Y: point of evaluation (list of int/float)
    
    this is required to construct the Hessian matrix
    for f = f(x1,x2,...,xN):
    [d2f/d(diff*x1), d2f/d(diff*x2), ..., d2f/d(diff*diff),..., d2f/d(diff*xN)]
    """
    
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    sGrad = []
    
    for k in range(0,len(X)):
        s = secondPartial(func,X,Y,diff1,X[k])
        sGrad.append(s)
    
    return sGrad

def Hessian(func,X,Y):
    """
    task: construct Hessian matrix
    input: objective function, func (str)
    X: list of variables (list of str)
    Y: point of evaluation (list of int/float)
    
    output: Hessian matrix: [[A11,A12..,A1N],[A21,A22,....,A2N]]
    """
    
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    hessian = []
    
    for k in range(0,len(X)):
        D = X[k]
        hessian.append(secondGrad(func,X,Y,D))
    
    return hessian

def inverseMatrix(M):
    """
    function: return inverse of matrix
    input: matrix; det(M) =/= 0
    output: inv(M)
    """
    
    H = np.array(M)
    Hinv = np.linalg.inv(H)
    
    return Hinv.tolist()
