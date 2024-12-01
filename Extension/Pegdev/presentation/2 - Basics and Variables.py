#Comment with a "#"


#Indentation
#Python is sensitive to indentation!
#Use TAB to jump some spaces


#Variables
#Names are case sensitive and cant start with numbers!
stringVar = "10"    #String
intVar = 10         #Int
floatVar = 10.1     #Float
booleanVar = True   #Boolean
print(stringVar, intVar, floatVar, booleanVar, "\n")


#Conversion and concat
print(intVar + floatVar)                #Int + Float        (It will sum the values)   
print(float(stringVar) + floatVar)      #String + Float     (It will sum the values)  
print(stringVar + str(floatVar))        #String + Float     (It will concat)
print(stringVar + stringVar, "\n")      #String + String    (It will concat)

stringVar = booleanVar
print(stringVar)
