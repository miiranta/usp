def f(x):
    return x*2.71828182845**x

def f1(x):
    return x*2.71828182845**x + 2.71828182845**x

def f2(x):
    return x*2.71828182845**x + 2*2.71828182845**x



def f1Prog(x, h):
    return ( -f(x+2*h) + 4*f(x+h) - 3*f(x) ) / (2*h)

def f1Center(x, h):
    return ( f(x+h) - f(x-h) ) / (2*h)

def f1Reg(x, h):
    return ( 3*f(x) - 4*f(x-h) + f(x-2*h) ) / (2*h)


def f2Prog(x, h):
    return ( 2*f(x) -5*f(x+h) + 4*f(x+2*h) - f(x+3*h) ) / h**2

def f2Center(x, h):
    return ( f(x+h) - 2*f(x) + f(x-h) ) / h**2

def f2Reg(x, h):
    return ( -f(x -3*h) + 4*f(x -2*h) - 5*f(x -h) + 2*f(x) ) / h**2


h = 0.1

print("F1 1.8 = ", f1Prog(1.8, h), f1(1.8), abs(f1Prog(1.8, h) - f1(1.8)))
print("F1 1.9 = ", f1Center(1.9, h), f1(1.9), abs(f1Center(1.9, h) - f1(1.9)))
print("F1 2.0 = ", f1Center(2.0, h), f1(2.0), abs(f1Center(2.0, h) - f1(2.0)))
print("F1 2.1 = ", f1Center(2.1, h), f1(2.1), abs(f1Center(2.1, h) - f1(2.1)))
print("F1 2.2 = ", f1Reg(2.2, h), f1(2.2), abs(f1Reg(2.2, h) - f1(2.2)))

print()

print("F2 1.8 = ", f2Prog(1.8, h), f2(1.8), abs(f2Prog(1.8, h) - f2(1.8)))
print("F2 1.9 = ", f2Center(1.9, h), f2(1.9), abs(f2Center(1.9, h) - f2(1.9)))
print("F2 2.0 = ", f2Center(2.0, h), f2(2.0), abs(f2Center(2.0, h) - f2(2.0)))
print("F2 2.1 = ", f2Center(2.1, h), f2(2.1), abs(f2Center(2.1, h) - f2(2.1)))
print("F2 2.2 = ", f2Reg(2.2, h), f2(2.2), abs(f2Reg(2.2, h) - f2(2.2)))