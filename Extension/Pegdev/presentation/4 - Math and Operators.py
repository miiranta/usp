#Math Operators
grade = input("What was your grade? \n")        #Always returns STRING!
grade = float(grade)                            #Converts whatever to float

grade = grade + 1
grade = grade - 0.5
grade = grade * 7
grade = grade // 3  #Returns INT
grade = grade / 6   #Returns FLOAT
grade = grade % 3   #Returns mod
grade = grade ** 2  #Returns Power

new_grade = ( grade - grade * 2 ** 4) * 3   #Follows math's priority rules 

print("Now it's", new_grade,";)")


#Assignment Operators
grade += 3 #grade = grade + 3
grade -= 3 #grade = grade - 3   #And basically every operator
print()


#Comparison Operators
print(1 == 2)
print(1 != 2)
print(1 >= 2)
print(1 <= 2)
print(1 > 2)
print(1 < 2)
print()


#Logical Operators
print(True and True)
print(False or True)
print(not False)
print(False or True and False or False)

