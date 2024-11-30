from math import *

def sec_line(x0, x1, f0, f1):
    """Retorna o valor da secante"""
    return (x0 * f1 - x1 * f0) / (f1 - f0)

def stop(x0, x1, epslon):
    """Retorna True se o intervalo de busca estiver dentro do epslon"""
    return epslon >= abs(x1 - x0) / abs(x1)

def make_function(function_string):
    def f(x):
        return eval(function_string + "-x")
    return f

x0 = float(input("Digite o minimo do intervalo: "))
x1 = float(input("Digite o máximo do intervalo: "))
epslon = float(input("Digite o epslon (tolerância): "))
function = make_function(input("Digite a função ex: cos(x) - x : "))

while not stop(x0, x1, epslon):
    "Ponto fixo para o método das secantes"
    x1, x0 = sec_line(x0, x1, function(x0), function(x1)), x1

print(f"Raíz mais próxima no intervalo dado: {x1}")