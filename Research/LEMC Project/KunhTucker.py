#importar bibliotecas e configurações iniciais
from sympy import *             # sin, cos, derivada, integral, limite, funções, infinito, latex, plot,  UM MONTE DE COISA
import matplotlib.pyplot as plt #construir gráficos
import numpy as np              #vetores e matrizes
import numbers                  #verifica se uma variável é um número
init_printing()                 #mostra em latex os outputs, não usaremos função print do python, roda apenas no notebook, não no console

#Define simbolos usados
x, y        = symbols('x, y')
LAM1, LAM2  = symbols('LAM1, LAM2,')

#Função f
f  = sympify("-(x - 2)**2 - (y - 3)**2")
f

#Restrição g1 
g1 = sympify("x - 1")
g1

#Restrição g2
g2 = sympify("y - 2")
g2

#Cria o lagrangeano
def criarLagrangeano(f, g1, g2):
  l = f - LAM1 * g1 - LAM2 * g2
  return l

#Verifica se os LAM são positivos nos candidatos, se sim, adiciona na lista candidatos
def verificarCandidato(solucao_sistema, candidatos):
  for candidato in solucao_sistema:
    
    #Tupla para lista (para editar)
    candidato = list(candidato)
    
    #Lista tamanho 2 para tamanho 4
    if(len(candidato) == 2):
      candidato.extend([0,0])
    
    #Algum LAM está indefinido? Mude para 0
    if (isinstance(candidato[2], numbers.Number) == False):
      candidato[2] = 0
    
    if (isinstance(candidato[3], numbers.Number) == False):
      candidato[3] = 0
    
    #Se os LAM são positivos, adiciona na lista
    if (candidato[2] >= 0 and candidato[3] >= 0):
      candidatos.append(candidato)
      print(candidato)

#Definindo alguns parametros antes da execução
l = criarLagrangeano(f, g1, g2)
lx = diff(l, x)
ly = diff(l, y)
g1x = diff(g1, x)
g1y = diff(g1, y)
g2x = diff(g2, x)
g2y = diff(g2, y)
candidatos = []


#TESTANDO OS CASOS (Lagrangeano)

#caso 1
#LAM1 >= 0 && LAM2>= 0
solucao_sistema = linsolve([lx,ly,g1,g2], (x, y, LAM1, LAM2))
verificarCandidato(solucao_sistema, candidatos)

#caso 2
#LAM1 >= 0 && LAM2 = 0
lx_aux = lx.subs(LAM2, 0)
ly_aux = ly.subs(LAM2, 0)
solucao_sistema = linsolve([lx_aux,ly_aux,g1], (x, y, LAM1, LAM2))
verificarCandidato(solucao_sistema, candidatos)

#caso 3
#LAM1 = 0 && LAM2 >= 0
lx_aux = lx.subs(LAM1, 0)
ly_aux = ly.subs(LAM1, 0)
solucao_sistema = linsolve([lx_aux,ly_aux,g2], (x, y, LAM1, LAM2))
verificarCandidato(solucao_sistema, candidatos)

#caso 4
#LAM1 = 0 && LAM2 = 0
lx_aux = lx.subs(LAM1, 0).subs(LAM2, 0)
ly_aux = ly.subs(LAM1, 0).subs(LAM2, 0)
solucao_sistema = linsolve([lx_aux,ly_aux], (x, y, LAM1, LAM2))
verificarCandidato(solucao_sistema, candidatos)


#TESTANDO OS CASOS (QR)

#caso 1
#Vg1  Vg2
solucao_sistema = linsolve([g1x, g1y, g2x, g2y], (x, y))
verificarCandidato(solucao_sistema, candidatos)

#caso 2
#Vg1 
solucao_sistema = linsolve([g1x, g1y], (x, y))
verificarCandidato(solucao_sistema, candidatos)

#caso 3
#Vg2
solucao_sistema = linsolve([g2x, g2y], (x, y))
verificarCandidato(solucao_sistema, candidatos)

#caso 4
#TRIVIAL


#AVALIANDO OS CANDIDATOS
resultadosDeF = []

for candidato in candidatos:
  resultadoDeF = f.subs(x, candidato[0]).subs(y, candidato[1]).subs(LAM1, candidato[2]).subs(LAM2, candidato[3])
  resultadosDeF.append(resultadoDeF)
  
  print(f"Candidato ({candidato[0]},{candidato[1]})", f" f({candidato[0]},{candidato[1]})= {resultadoDeF}")
  
#Encontrando o candidato com maior/menor valor de f
valor_max = max(resultadosDeF)
index_max = resultadosDeF.index(valor_max)

valor_min = min(resultadosDeF)
index_min = resultadosDeF.index(valor_min)

print(f"\nO candidato com maior valor de f é ({candidatos[index_max][0]},{candidatos[index_max][1]}) com valor de f = {valor_max}")
print(f"O candidato com menor valor de f é ({candidatos[index_min][0]},{candidatos[index_min][1]}) com valor de f = {valor_min}")