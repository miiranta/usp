# Problema 8 Puzzle

# 1 2 3
# 4 5 6
# 7 8
# [1,2,3,4,5,6,7,8,0]

import sys
sys.setrecursionlimit(10000)

#Problema a)
initial_state   = [8,1,3,7,0,2,6,5,4]
final_state     = [1,2,3,8,0,4,7,6,5]

#Problema b)
#Não resolve na mesma ordem do prolog aparentemente
#Precisa de muita MUITA memória mesmo pra resolver
#initial_state   = [5,4,0,6,1,8,7,3,2]
#final_state     = [1,2,3,8,0,4,7,6,5]

def up(initx):
    init = initx.copy()
    
    pos = init.index(0)
    if pos not in [0,1,2]:
        init[pos], init[pos-3] = init[pos-3], init[pos]
    return init
    
def down(initx):
    init = initx.copy()
    
    pos = init.index(0)
    if pos not in [6,7,8]:
        init[pos], init[pos+3] = init[pos+3], init[pos]
    return init
    
def left(initx):
    init = initx.copy()
    
    pos = init.index(0)
    if pos not in [0,3,6]:
        init[pos], init[pos-1] = init[pos-1], init[pos]
    return init
    
def right(initx):
    init = initx.copy()
    
    pos = init.index(0)
    if pos not in [2,5,8]:
        init[pos], init[pos+1] = init[pos+1], init[pos]
    return init
    
def isInPath(move, path):
    for i in path:
        if move == i:
            return True
    return False
    
def recursion(init, final, path, moves, stop):
    queue = []
    
    #Solução trivial
    if init == final:
        return path, moves, stop

    mr = right(init)
    ml = left(init)
    mu = up(init)
    md = down(init)
    
    if not isInPath(ml, path):
        queue.append(ml)
        moves.append("Left")
        
    if not isInPath(mr, path):
        queue.append(mr)    
        moves.append("Right")
        
    if not isInPath(mu, path):
        queue.append(mu)  
        moves.append("Up")    
        
    if not isInPath(md, path):
        queue.append(md)
        moves.append("Down") 
     
    for i in queue:
        newpath = path.copy()
        newpath.append(i)
        
        if(newpath[-1] == final):
            stop = True
            return newpath, moves, stop
        
        if(not(stop)):
            path, moves, stop = recursion(i, final, newpath, moves, stop)

    return path, moves, stop
     
def solve(initial_state, final_state):
    path, moves, stop = recursion(initial_state, final_state, [initial_state], [], False)
    return path, moves       
        

#Resolvendo
path, moves = solve(initial_state, final_state)

print("\nCaminho: ")
print(path)

print("\nMovimentações necessárias: ", len(path))
print(moves)
