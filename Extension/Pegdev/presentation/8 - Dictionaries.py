#Dictionaries
#Can be changed

dic1 = dict()
dic2 = {
    "good":"indiana jones",
    "bad":"emoji movie",
    1:"Hey",
    2:"Bye"
}

dic2[3] = "Yall"
dic2["amazing"] = "wonderful"
del dic2[1]

print(dic2)
print(dic2.values())    #Without keys
print(dic2[3])
