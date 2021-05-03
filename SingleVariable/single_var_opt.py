"""
SINGLE VARIABLE OPTIMIZATION ALGORITHMS:
1. GRADIENT DESCENT
2. GRADIENT ASCENT
3. INTERVAL SEARCH
4. SWANN'S BRACKET
5. POWELL'S METHOD

CREDITS:
http://home.cc.umanitoba.ca/~lovetrij/cECE7670/Files/optim1.pdf
"""

from derivative import derivative_single
from derivative import outputfunc
from derivative import negativeOut
import numpy as np
import scipy

def gradDescent_single(func,a,b,guess,var):
    """
    task: computes optimum (X,Y) of Y = f(X) using gradient descent numerical method
    func: objective function of interest (str)
    guess: initial guess for optimization (int)
    var: variable to differentiate (str)
    
    OUTPUT: (X,Y) (dict) that gives optimized Y in f(X)
    """
    a=0
    b=0
    alpha = 0.001
    xf = guess
    e = 1
    
    while e > 1e-12:
        dfunc = derivative_single(func,xf,var)
        x_n = xf - alpha*dfunc
        e = abs((x_n-xf)/(x_n))
        xf = x_n
    
    y_val = outputfunc(func,xf,'x')
    
    if abs(xf-guess) > 10000:
        raise ValueError("Guess causes divergence, or there is no local minima.")
    else:
        return {'x coordinate: ': xf, 'y coordinate: ': y_val}

def gradAscent_single(func,a,b,guess,var):
    """
    task: computes optimum (X,Y) of Y = f(X) using gradient ascent numerical method
    func: objective function of interest (str)
    guess: initial guess for optimization (int)
    var: variable to differentiate (str)
    
    OUTPUT: (X,Y) (dict) that gives optimized Y in f(X)
    """
    a=0
    b=0
    alpha = 0.001
    xf = guess
    e = 1
    
    while e > 1e-12:
        dfunc = derivative_single(func,xf,var)
        x_n = xf + alpha*dfunc
        e = abs((x_n-xf)/(x_n))
        xf = x_n
    
    y_val = outputfunc(func,xf,var)
    
    if abs(xf-guess) > 10000:
        raise ValueError("Guess causes divergence, or there is no local maxima.")
    
    else:
        return {'x coordinate: ': xf, 'y coordinate: ': y_val}

def gradAll(func,a,b,guess,var,option = None):
    """
    returns the coordinates of the maxima and minima for Y = f(X)
    input: func (str), {a = 0, b = 0}
    guess = initial guess for X (int)
    var = variable to differentiate (str)
    DEFAULT SETTING: compute minima
    """
    min_coord = None
    max_coord = None
    
    if option == None:
        min_coord = gradDescent_single(func,a,b,guess,var)
    elif option == 'maxima':
        max_coord = gradAscent_single(func,a,b,guess,var)
            
    return {'minima': min_coord, 'maxima': max_coord}

def intervalSearch(func,a,b,guess,var):
    """
    task: compute optimum (X,Y) of Y = f(X) using interval search method
    func: objective function of interest (str)
    a: lower bound of interval guess
    b: upper bound of interval guess
    var: variable to differentiate of interest (str)
    
    output: (X,Y) (dict) that shows where optimum value is located
    """
    guess=0
    delta = b-a
    xm = a+(delta/2)
    fm = outputfunc(func,xm,var)
    
    while delta > 1e-12:
        xL = a+delta/4
        fL = outputfunc(func,xL,var)
        xu = b-delta/4
        fu = outputfunc(func,xu,var)
        
        if fL < fm:
            b = xm
            xm = xL
            fm = fL
            delta = b-a
        elif fu < fm:
            a = xm
            xm = xu
            fm = fu
            delta = b-a
        else:
            a = xL
            b = xu
            delta = b-a
        
    interval = (a,b)
    y_interval = outputfunc(func,a,var)
    
    return {'x coordinate: ': a, 'y coordinate: ': y_interval}


def SwannsBracket(func,a,b,xo,var,option = None):
    """
    task: utilizes Swanns Bracket algorithm to find global maxima/minima of function
    note: accounts for non-unimodal functions; error raised if non-unimodal
    input: f = f(X) (str)
    xo = x-coordinate of interest (int)
    var = variable to differentiate of interest (str)
    """
    
    if option == None:
        a=0
        b=0
        delta = 0.001
        xl = xo-delta
        xu = xo+delta
        fL = outputfunc(func,xl,var)
        fu = outputfunc(func,xu,var)
        fo = outputfunc(func,xo,var)
        i = 1

        if (fo >= fu) and (fo <= fL):
            while fu < fo:
                i = i + 1
                xl = xo
                xo = xu
                fL = fo
                fo = fu

                xu = xu + (2**i)*delta
                fu = outputfunc(func,xu,var)
            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo>=fL) and (fo <= fu):
            while fL<fo:
                xu = xo
                xo = xl
                fu = fo
                fo = fL
                xl = xl-(2**i)*delta
                fL = outputfunc(func,xl,var)

            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo<=fu) and (fo<=fL):
            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo>=fu) and (fo>=fL):
            raise ValueError("cannot be computed as this is a non-unimodal function")

    elif option == 'maxima':
        
        a=0
        b=0
        delta = 0.001
        xl = xo-delta
        xu = xo+delta
        fL = negativeOut(func,xl,var)
        fu = negativeOut(func,xu,var)
        fo = negativeOut(func,xo,var)
        i = 1

        if (fo >= fu) and (fo <= fL):
            while fu < fo:
                i = i + 1
                xl = xo
                xo = xu
                fL = fo
                fo = fu

                xu = xu + (2**i)*delta
                fu = negativeOut(func,xu,var)
            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo>=fL) and (fo <= fu):
            while fL<fo:
                xu = xo
                xo = xl
                fu = fo
                fo = fL
                xl = xl-(2**i)*delta
                fL = negativeOut(func,xl,var)

            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo<=fu) and (fo<=fL):
            return {'x coordinate: ': xl, 'y coordinate: ': outputfunc(func,xl,var)}

        elif (fo>=fu) and (fo>=fL):
            raise ValueError("cannot be computed as this is a non-unimodal function")


def PowellMethod(func,a1,delta,del_max,var):
    """
    task: utilize Powell's Method (algorithm) to find local minima of function f = f(X)
    inputs: {func, a1, delta, del_max, var}
    func: objective function of interest (str)
    a1: starting point for guess (int)
    delta: difference between a_k and a_(k+1) (int)
    del_max: the maximum difference between a -> b (int)
    var: variable to differentiate of interest (str)
    """
    
    c1 = a1 + delta
    f_a = outputfunc(func,a1,var)
    f_c = outputfunc(func,c1,var)
    x = 0
    a2 = 0
    b2 = 0
    c2 = 0
    fwd = False
    
    if f_a > f_c:
        b1 = a1 + 2*delta
        f_b = outputfunc(func,b1,var)
        fwd = True
    else:
        b1 = c1
        c1 = a1
        a1 = a1 - delta
        f_b = f_c
        f_c = f_a
        f_a = outputfunc(func,a1,var)
    
    while (f_c < f_a) and (f_c < f_b):
        p = ((c1-b1)*f_a + (a1-c1)*f_b + (b1-a1)*f_c)/((b1-c1)*(c1-a1)*(a1-b1))
        
        if p > 0:
            x = (1/2)*((b1**2 - c1**2)*f_a + (c1**2 - a1**2)*f_b + (a1**2 - b1**2)*f_c)/((b1-c1)*f_a + (c1-a1)*f_b + (a1-b1)*f_c)
        
        if fwd == True:
            if p <= 0:
                a2 = a1
                b2 = b1+del_max
                c2 = c1
                f_b = outputfunc(func,b2,var)
            else:
                if (x-b1) > del_max:
                    b2 = b1 + del_max
                else:
                    b2 = x
                
                a2 = c1
                c2 = b1
                f_a = f_c
                f_c = f_b
                f_b = outputfunc(func,b2,var)
        else:
            if p <= 0:
                a2 = a1 - del_max
                b2 = b1
                c2 = c1
                f_a = outputfunc(func,a2,var)
            else:
                if (a1-x) > del_max:
                    a2 = a1 - del_max
                else:
                    a2 = x
            b2 = c1
            c2 = a1
            f_b = f_c
            f_c = f_a
            f_a = outputfunc(func,a2,var)
     
    return {'x coordinate: ': b2, 'y-coordinate: ': f_a}