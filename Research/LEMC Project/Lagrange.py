#importar bibliotecas e configurações iniciais
from sympy import *             # sin, cos, derivada, integral, limite, funções, infinito, latex, plot,  UM MONTE DE COISA
import matplotlib.pyplot as plt #construir gráficos
import numpy as np              #vetores e matrizes
init_printing()                 #mostra em latex os outputs, não usaremos função print do python, roda apenas no notebook, não no console

#Define simbolos usados
w, x, y, z = symbols('w, x, y, z')
a1, a2, a3, a4 = symbols('a1, a2, a3, a4')

#Define função a ser resolvida
f = sympify("x**2 + 2*y**2-x*y")

#Define restrições
g1 = sympify("x+y-8")

#Cria o lagrangeano
def criarLagrangeano(f,g1):
    l = f - a1 * g1
    return l

#Calcular o gradiente (x,y)
def gradiente(function):
    df_x    =  diff(f, x)
    df_y    =  diff(f, y)
    return [df_x, df_y]

#Calcula o hessiano
def calcularHessiano(candidato, l):
    lh = l.subs(a1, candidato[2])
    detHessian = hessian(lh, (x, y)).det()
    
    if(detHessian > 0):
        print("tem Hessiano positivo igual a", detHessian) 
        return true
        
    if(detHessian < 0):
        print("tem Hessiano negativo igual a", detHessian, "e portanto é ponto de sela.")
        return false
    
    if(detHessian == 0):
        print("tem Hessiano igual a 0 e portanto é indeterminado.")
        return false
    
#Testa se os candidatos são concavos ou convexos (MAX ou MIN)
def testarConvexidade(candidato, lx2):
    lx_aux = lx2.subs(x, candidato[0]).subs(y, candidato[1]).subs(a1, candidato[2])
    
    if(lx_aux < 0):
        print ("e é ponto MAX")
    if(lx_aux > 0):
        print("e é ponto MIN")
    if(lx_aux == 0):
        print("e é indeterminado")
    
    return lx2

l = criarLagrangeano(f,g1)
lx = diff(l,x)
ly = diff(l,y)
lx2 = diff(lx,x)

#Resolvendo um sistema linear (Algebra linear)
candidatos = nonlinsolve([lx,ly,g1], (x, y, a1))
print("Os candidatos são:", candidatos, "\n")

for candidato in candidatos:
        print("O candidato", candidato)
        if(calcularHessiano(candidato, l)):
            testarConvexidade(candidato, lx2)
        print("\n")
    

