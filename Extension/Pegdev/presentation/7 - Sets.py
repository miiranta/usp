#Sets
#Can´t have multiple occurencies of the same object
#Works with loops

set1 = set()
set2 = set(["Hello", "Goodbye"])
set3 = frozenset()
print(set2)


set2.add("Hello") #Wont make any difference
print(set2)

set2.add("Good Morning")
print(set2)

set2.remove("Hello")
#also try set2.discard("Hello")
print(set2)

set2.clear()
print(set2)


#Some functions
set1.union(set2)
set1.intersection(set2)
set1.difference(set2)
set1.issubset(set2)
set1.isdisjoint(set2)

