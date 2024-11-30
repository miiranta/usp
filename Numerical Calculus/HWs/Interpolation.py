from random import *
from sympy import *     
from decimal import *     
import numpy as np
import time

start_time = time.time()

points = []

def addPoint(x, y):
    points.append((x, y))

def creatMatrixA(vec):
    vecLen = len(vec)
    mat = np.ones((vecLen, vecLen))
    i = 0
    
    for point in vec:
        for j in range(vecLen - 1):
            mat[i][j] = Decimal(point[0]) ** Decimal(vecLen - j - 1)
        i = i + 1
    
    return mat

def creatMatrixB(vec):
    vecLen = len(vec)
    mat = np.ones((vecLen))
    i = 0
    
    for point in vec:
        mat[i] = Decimal(point[1])
        i = i + 1
    
    return mat
    
def findPoly(points):
    A = creatMatrixA(points)
    b = creatMatrixB(points)
    sol = np.linalg.solve(A, b)
    
    i = 0
    solLen = len(sol)
    for item in sol:
        print( "+ (" + str(item) + ")*",  end = "")
        print("x**" + str(solLen - i - 1) + " ", end = "")
        
        i = i + 1

    return sol


for j in range(100):
    addPoint(j, randint(0, 100))


#print(points)
print("\n")
findPoly(points)


print("\n\n--- %s seconds ---" % (time.time() - start_time))