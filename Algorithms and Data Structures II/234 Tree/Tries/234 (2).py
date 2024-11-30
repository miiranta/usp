#Classe nó
class Node:
    
    def __init__(self):
        self.parent = None
        self.children = [None, None, None, None] #Maximo: 4
        self.keys = [None, None, None] #Maximo: 3

class Tree234:
    
    def __init__(self):
        self.root = None
        
    def insert(self, key):
        self.insert_recursive(self.root, key)
        print("Inserindo: ", key)
        
    def insert_recursive(self, node, key):
        
        if(node != None):
            keys_num = self.list_size(node.keys)
            
        else:
            node = Node()
            node.keys.append(key)
            node.keys = self.list_sort(node.keys)
            
            if(self.root == None):
                self.root = node
            
            return node
        
        if(self.is_leaf(node)):
        
            if keys_num == 0:
                node.keys.append(key)
                node.keys = self.list_sort(node.keys)
                return node
        
            elif keys_num == 1:
                node.keys.append(key)
                node.keys = self.list_sort(node.keys)
                return node

            elif keys_num == 2:
                node.keys.append(key)
                node.keys = self.list_sort(node.keys)
                return node

        if keys_num == 3:
            new_node = self.split(node)
            self.insert_recursive(new_node, key)
            return node
            
        if (keys_num > 0 and key < node.keys[0]):
            node.children[0] = self.insert_recursive(node.children[0], key)
            if(node.children[0] != None):
                node.children[0].parent = node
                 
        elif (node.keys[1] == None) or (keys_num > 1 and key < node.keys[1]):
            node.children[1] = self.insert_recursive(node.children[1], key)
            if(node.children[1] != None):
                node.children[1].parent = node
        
        elif (node.keys[2] == None) or (keys_num > 2 and key < node.keys[2]):
            node.children[2] = self.insert_recursive(node.children[2], key)
            if(node.children[2] != None):
                node.children[2].parent = node
        
        else:
            node.children[3] = self.insert_recursive(node.children[3], key)
            if(node.children[3] != None):
                node.children[3].parent = node
                
        return node
 
    def delete(self, key):
        self.delete_recursive(self.root, key)
        print("Deletando: ", key)
        
    def delete_recursive(self, node, key):
        print("Oi")
        
    def is_leaf(self, node):
        return node.children[0] == None and node.children[1] == None and node.children[2] == None and node.children[3] == None
    
    def organize_children(self, node):
        sorted_children = self.list_sort_children(node.children)
        
        node.keys = self.list_sort(node.keys)
        node.children = [None, None, None, None]
        
        for i in range(len(sorted_children)):
            for j in range(len(node.keys)):
                last = j
                
                if(node.keys[j] != None):
                    if(sorted_children[i] != None):
                        
                        if(sorted_children[i].keys[0] < node.keys[j]):
                            node.children[j] = sorted_children[i]
                            sorted_children[i] = None

        for i in range(len(sorted_children)):
            if(sorted_children[i] != None):
                node.children[last] = sorted_children[i]
                sorted_children[i] = None

        return node.children              
      
    def list_sort_children(self, list):
        child_list = []
        child_list_keys = []
        sorted_child_list = []
        
        for child in list:
            if child != None:
                child_list.append(child)
                child_list_keys.append(child.keys[0])
        
        child_list_keys = sorted(child_list_keys)
  
        for key in child_list_keys:
            for child in child_list:
                if child.keys[0] == key:
                    sorted_child_list.append(child)

        for i in range(4 - len(sorted_child_list)):
            sorted_child_list.append(None)
  
        return sorted_child_list
          
    def list_sort(self, list):
        filtered_list = filter(None, list)
        sorted_list = sorted(filtered_list)
        
        for i in range(3 - len(sorted_list)):
            sorted_list.append(None)
        
        return sorted_list
    
    def list_size(self, list):
        return sum(x is not None for x in list)
    
    def split(self, node):
            
        if node != None:
            keys_num = self.list_size(node.keys)
                
        if keys_num == 3:

            if node == self.root:
                
                self.root = Node()
                self.root.keys[0] = node.keys[1]
                
                node1 = Node()
                node1.keys[0] = node.keys[0]
                node1.children[0] = node.children[0]
                node1.children[1] = node.children[1]
                node1.parent = self.root
                self.root.children[0] = node1
                
                node2 = Node()
                node2.keys[0] = node.keys[2]
                node2.children[0] = node.children[2]
                node2.children[1] = node.children[3]
                node2.parent = self.root
                self.root.children[1] = node2
                
                node = self.root
                
                #print("Splitting")
                #self.print()
                
                return node
            
            else:
                node.parent.keys.append(node.keys[1])
                node.parent.keys = self.list_sort(node.parent.keys)
                node.keys[1] = None
            
                node.keys = self.list_sort(node.keys)
                
                node.parent.children.remove(node)
                node.parent.children.append(None)
                
                node1 = Node()
                node1.keys.append(node.keys[0])
                node1.keys = self.list_sort(node1.keys)
                node.parent.children.append(node1)
                node1.parent = node.parent
                node1.children[0] = node.children[0]
                node1.children[1] = node.children[1]
                
                node2 = Node()
                node2.keys.append(node.keys[1])
                node2.keys = self.list_sort(node2.keys)
                node.parent.children.append(node2)
                node2.parent = node.parent
                node2.children[0] = node.children[2]
                node2.children[1] = node.children[3]
                
                node.parent.children = self.organize_children(node.parent)
                
                #print("Splitting")
                #self.print()
                
                return node.parent
        else:
            
           
            
            return node
                     
    def print(self):
        print("R-------------------")
        self.print_recursive(self.root, 0)
        print("L-------------------")
    
    def print_recursive(self, node, level):
        
        if node != None:
 
            keys_num = self.list_size(node.keys)    
            
            print("       "*level,"--")
            
            if node.children[0] != None:
                self.print_recursive(node.children[0], level+1)
            
            if(keys_num > 0):
                print("       "*level, node.keys[0])
            else:
                print("       "*level, "None")
                
            if node.children[1] != None:
                self.print_recursive(node.children[1], level+1)
            
            if(keys_num > 1):
                print("       "*level, node.keys[1])
            else:
                print("       "*level, "None")

            if node.children[2] != None:
                self.print_recursive(node.children[2], level+1)
                
            if(keys_num > 2):
                print("       "*level, node.keys[2])
            else:
                print("       "*level, "None")

            if node.children[3] != None:
                self.print_recursive(node.children[3], level+1)

            # if(node.parent != None):
            #     print("       "*level,">Parent: ", node.parent.keys[0])

            print("       "*level,"--")



arvore = Tree234()

arvore.insert(7)
arvore.print()

arvore.insert(3)
arvore.print()

arvore.insert(2)
arvore.print()

arvore.insert(4)
arvore.print()

arvore.insert(10)
arvore.print()

arvore.insert(12)
arvore.print()

arvore.insert(11)
arvore.print()

arvore.insert(13)
arvore.print()

arvore.insert(14)
arvore.print()

arvore.insert(6)
arvore.print()




