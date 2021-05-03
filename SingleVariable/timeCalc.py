import time
from single_var_opt import gradDescent_single
from single_var_opt import gradAscent_single
from single_var_opt import intervalSearch
from single_var_opt import SwannsBracket

def timeCalc(func,a,b,xo,var):
    """
    task: calculates time required to run numerical optimization
    output form: {'method name':[time, numericalOutput()]}
    input: func (str), a,b (int, depends on method used), xo (int), var (str)    
    """
    
    l = [gradDescent_single,intervalSearch,SwannsBracket]
    t = {}
    c = 0
    
    for i in l:
        start = time.time()
        c = i(func,a,b,xo,var)
        end = time.time()
        t1 = (end - start)
        t[i.__name__] = [t1,c]
    
    return t