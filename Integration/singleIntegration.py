from math import *

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

def LeftHandRule(func,a,b,var_name,dx = 0.00001):
    """
    uses left-hand rule to integrate a single variable function f = f(X) from a≤X≤b
    input: function of interest (str), a (lower bound), b (upper bound)
    output: approximation of definite integral of f(X) from a to b
    """
    a_update = a
    LHR = 0
    
    if a == b:
        return 0
    elif a < b:
        while a_update < (b-dx):
            LHR = LHR + (outputfunc(func,a_update,var_name)*dx)
            a_update = a_update + dx
        
        return LHR
    elif b < a:
        while a_update < (b-dx):
            LHR = LHR + (outputfunc(func,a_update,var_name)*dx)
            a_update = a_update + dx
            
        return -1*LHR

def RightHandRule(func,a,b,var_name,dx = 0.00001):
    """
    uses right-hand rule to integrate a single variable function f = f(X) from a≤X≤b
    input: function of interest (str), a (lower bound), b (upper bound)
    output: approximation of definite integral of f(X) from a to b
    """
    a_update = a + dx
    RHR = 0
    
    if a == b:
        return 0
    elif a < b:
        while a_update < b:
            RHR = RHR + (outputfunc(func,a_update,var_name)*dx)
            a_update = a_update + dx
        
        return RHR
    elif b < a:
        while a_update < b:
            RHR = RHR + (outputfunc(func,a_update,var_name)*dx)
            a_update = a_update + dx
        
        return -1*RHR

def Midpoint(func,a,b,var_name,n = 20000):
    """
    uses midpoint rule to integrate a single variable function f = f(X) from a≤X≤b
    input: function of interest (str), a (lower bound), b (upper bound)
    output: approximation of definite integral of f(X) from a to b
    """
    h = (b-a)/n
    c = 0
    MP = 0
    xi = a
    
    if a == b:
        return 0
    elif a < b:
        while c < (n-1):
            x1 = xi + h
            x2 = (xi + x1)/2
            MP = MP + (outputfunc(func,x2,var_name)*h)
            xi = xi + h
            c = c + 1
        
        return MP
    elif b < a:
        while c < (n-1):
            x1 = xi + h
            x2 = (xi + x1)/2
            MP = MP + (outputfunc(func,x2,var_name)*h)
            xi = xi + h
            c = c + 1
        
        return -1*MP

def Trapezoid(func,a,b,var_name,n = 20000):
    """
    uses trapezoid rule to integrate a single variable function f = f(X) from a≤X≤b
    input: function of interest (str), a (lower bound), b (upper bound)
    output: approximation of definite integral of f(X) from a to b
    """
    h = (b-a)/n
    c = 0
    T = 0
    xi = a
    yi = outputfunc(func,xi,var_name)
    
    if a == b:
        return 0
    elif a < b:
        while c < n:
            yi = outputfunc(func,xi,var_name)
            x1 = xi + h
            y1 = outputfunc(func,x1,var_name)
            p = ((yi + y1)/2)*h
            T = T + p
            xi = xi + h
            c = c + 1
        
        return T
    elif b < a:
        while c < n:
            x1 = xi + h
            y1 = outputfunc(func,x1,var_name)
            p = ((yi + y1)/2)*h
            T = T + p
            xi = xi + h
            c = c + 1
        
        return -1*T

def Simpson(func,a,b,var_name,n = 20000):
    """
    uses simpson's rule to integrate a single variable function f = f(X) from a≤X≤b
    input: function of interest (str), a (lower bound), b (upper bound)
    output: approximation of definite integral of f(X) from a to b
    """
    h = (b-a)/n
    c = 0
    S = 0
    xi = a
    
    if a == b:
        return 0
    elif a < b:
        while c < n:
            yi = outputfunc(func,xi,var_name)
            y1 = 4*outputfunc(func,(xi+0.5*h),var_name)
            y2 = outputfunc(func,(xi+h),var_name)
            p = h*(yi+y1+y2)
            S = S + p
            xi = xi + h
            c = c + 1
        
        return (1/6)*S
    elif b < a:
        while c < n:
            yi = outputfunc(func,xi,var_name)
            y1 = 4*outputfunc(func,(xi+0.5*h),var_name)
            y2 = outputfunc(func,(xi+h),var_name)
            p = h*(yi+y1+y2)
            S = S + p
            xi = xi + h
            c = c + 1
        
        return (-1/6)*S