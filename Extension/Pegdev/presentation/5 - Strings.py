#Strings
queijadinha = "gooD"

print(queijadinha)
print(queijadinha.upper())          #Note: the methods does not change the original string, it returns a new one
print(queijadinha.lower())
print(queijadinha.capitalize())
print(queijadinha.replace("oD", "odelicious")) 
print(5 * queijadinha, "\n")


#IN operator
print(queijadinha.find("D"))        #Index of D in the string
print("oo" in queijadinha)          #Is there "oo" in the string? True/False
