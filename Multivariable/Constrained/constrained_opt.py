from pderivative import partialDerivative
from pderivative import multiOut
from pderivative import grad
from multivar_optimization import multiGrad
import math

def multiGradConstraint(func,X,Y,constraint,option = None):
    """
    task: calculate the optimum point for a multivariable function: F = F(x1,x2,...,xn)
    func: objective function (str)
    X: list of variables (strings) {x1,x2,...,xn}
    Y: initial guess for the optimum value (list of int/floats)
    constraint: [[x1,x2],[y1,y2]]
    option: choose whether you want to solve maxima or minima. default option: minima
    
    output: {solution, mean error} (dict)
    """
    solutionC = list()
    extrema = list()
    c = None
    criteria = True
    
    if (option == None) or (option.lower() == 'minima'):
        solution = multiGrad(func,X,Y)['optimum solution: ']
        
        if len(constraint) == len(solution):
            for k in range(0,len(constraint)):
                if (solution[k] >= constraint[k][0]) and (solution[k] <= constraint[k][1]):
                    solutionC.append(solution[k])
                else:
                    criteria = False
                    
        else:
            raise ValueError("dim(constraint) is not equal to dim(solution)")
    
    if (option.lower() == 'maxima'):
        solution = multiGrad(func,X,Y,option = 'maxima')['optimum solution: ']
        
        if len(constraint) == len(solution):
            for k in range(0,len(constraint)):
                if (solution[k] >= constraint[k][0]) and (solution[k] <= constraint[k][1]):
                    solutionC.append(solution[k])
                else:
                    criteria = False
        
        else:
            raise ValueError("dim(constraint) is not equal to dim(solution)")
    
    if (criteria == False) and (option == None or option.lower() == 'minima'):
        print("There is no local minima within set of constraints established.")
    
    elif (criteria == False) and (option.lower() == 'maxima'):
        print("There is no local maxima within set of constraints established.")
    
    elif (criteria == True):
        return solutionC