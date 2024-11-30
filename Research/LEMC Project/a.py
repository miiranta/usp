def function_generator(function_string):
    def f(x):
        return eval(function_string)
    return f

f = function_generator("(-0.9583333333333334)*x**2 + (4.083333333333334)*x**1 + (19.0)*x**0")

print(f(-2))
print(f(4))
print(f(6))



