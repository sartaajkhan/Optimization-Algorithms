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
    seg1 = eval(func, {diff: x+h})
    seg2 = eval(func, {diff: x})
    dydx = (seg1-seg2)/h
    
    return dydx


def outputfunc(func,x,var_name):
    """
    task: computes function
    func: function of interest (str)
    x: variable substituted into function (int/float)
    var_name: variable name to be substituted in (str)
    """
    
    y = eval(func, {var_name: x})
    return y

def negativeOut(func,x,var_name):
    """
    task: computes function but multiplies by -1
    func: function of interest (str)
    x: variable substituted into function (int/float)
    var_name: variable name to be substituted in (str)
    """
    
    y = -1*eval(func, {var_name: x})
    return y