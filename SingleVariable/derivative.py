import numpy
from scipy.misc import derivative
import math


def derivative_single(func,x,diff):
    """
    task: numerically computes the derivative at a certain point x = xo for single variable function
    inputs: func (str), x (int), diff: variable to differentiate (str)
    output: approximate derivative at point X = Xo (numerical)
    
    """
    h = 1e-10
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    seg1 = eval(func, {diff: x+h}, math_ops)
    seg2 = eval(func, {diff: x}, math_ops)
    dydx = (seg1-seg2)/h
    
    return dydx


def outputfunc(func,x,var_name):
    """
    task: computes function
    func: function of interest (str)
    x: variable substituted into function (int/float)
    var_name: variable name to be substituted in (str)
    """
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    y = eval(func, {var_name: x}, math_ops)
    return y

def negativeOut(func,x,var_name):
    """
    task: computes function but multiplies by -1
    func: function of interest (str)
    x: variable substituted into function (int/float)
    var_name: variable name to be substituted in (str)
    """
    math_ops = {'sqrt':sqrt,'log':log,'sin':sin,'cos':cos,'tan':tan,
                'asin':asin, 'acos':acos, 'atan':atan, 'asinh':asinh, 'acosh':acosh,
                'atanh':atanh, 'pow':pow, 'exp':exp,
                'fabs':fabs, 'factorial': factorial, 'floor': floor}
    
    y = -1*eval(func, {var_name: x}, math_ops)
    return y
