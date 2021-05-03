from derivative import derivative_single
from derivative import outputfunc
from derivative import negativeOut


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


def gradCalc(func,var,lb,ub):
    """
    task: able to compute and return all local maxima and minima within a specified interval
    lb: lower bound of interval (int)
    ub: upper bound of interval (int)
    var: variable to differentiate of interest (str)
    func: objective function (str)
    """
    a1 = lb
    delta = 1
    b1 = lb+0.001
    min_coord = list()
    max_coord = list()
    exist_min = list()
    exist_max = list()
    a = 0
    b = 0
    
    while b1 < ub:
        a1 = a1 + delta
        b1 = a1 + delta
        for i in [a1,b1]:
            dfunc = derivative_single(func,a1,var)
            if dfunc < 0:
                try:
                    min_coord.append(gradDescent_single(func,a,b,b1,var))
                    
                except ValueError:
                    pass
            elif dfunc > 0:
                try:
                    max_coord.append(gradAscent_single(func,a,b,b1,var))
                except ValueError:
                    pass
    
    min_coord_exist = list()
    min_coord_new = list()
    for i in min_coord:
        if round(i['x coordinate: '],3) in min_coord_exist:
            pass
        else:
            min_coord_exist.append(round(i['x coordinate: '],3))
            min_coord_new.append({'x coordinate: ':i['x coordinate: '], 'y coordinate: ':i['y coordinate: ']})
    
    max_coord_exist = list()
    max_coord_new = list()
    
    for i in max_coord:
        if round(i['x coordinate: '],3) in max_coord_exist:
            pass
        else:
            max_coord_exist.append(round(i['x coordinate: '],3))
            max_coord_new.append({'x coordinate: ':i['x coordinate: '], 'y coordinate: ':i['y coordinate: ']})
    
    return {'minima': min_coord_new, 'maxima': max_coord_new}