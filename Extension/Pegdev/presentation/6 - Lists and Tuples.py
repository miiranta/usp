#Lists
song = ["We", "All", "Live", "In a", "Yellow", "Submarine"]
print(song)
print(song[0])
print(song[1])
print(song[-1]) #Last element
print(song[-2], "\n") 

song[0] = "You"
print(song[0], "\n")

print(song[0:3], "\n")  #All elements from 0 to 2


#Methods for lists
song.append("In the Ocean") #Modifies original object!
print(song)

song.insert(5, "Cool")
print(song)

song.remove("Cool")
print(song)

truth = "In the Ocean" in song
print(truth)

print(len(song))

song.clear()
print(song)


#Tuples
numbers = (3,2,1)   #Not possible to change
