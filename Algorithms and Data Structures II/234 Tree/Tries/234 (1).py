class Node():
    
    child1 = None
    key1 = None
    child2 = None
    key2 = None
    child3 = None
    key3 = None
    child4 = None
    
class Tree234():
    
    def __init__(self):
        self.root = Node()
    
    def insert(self, key):
        self.insert_recursive(self.root, key)
        
    def insert_recursive(self, node, key):
        if node == None:
            node = Node()
            node.key1 = key
        elif node.key1 == None:
            node.key1 = key
        elif node.key2 == None:
            node.key2 = key
        elif node.key3 == None:
            node.key3 = key
            self.split(node)
        else:
            if key < node.key1:
                print("oi")
                self.insert_recursive(node.child1, key)
            elif key < node.key2:
                print("oi2")
                self.insert_recursive(node.child2, key)
            elif key < node.key3:
                print("oi3")
                self.insert_recursive(node.child3, key)
            else:
                print("oi4")
                self.insert_recursive(node.child4, key)
            
        
    def split(self, node):
        print("Split")
    
    def delete(self, key):
        print("Delete")
        
    def seach(self, key):
        print("Search")

    def print(self):
        self.print_recursive(self.root)
    
    def print_recursive(self, node):
        if node != None:
            self.print_recursive(node.child1)
            print(node.key1, end=" ")
            self.print_recursive(node.child2)
            print(node.key2, end=" ")
            self.print_recursive(node.child3)
            print(node.key3, end=" ")
            self.print_recursive(node.child4)    
    
    
    
arvore = Tree234()
arvore.insert(1)
arvore.insert(20)
arvore.insert(3)
arvore.insert(10)
arvore.insert(19)
arvore.insert(32)
arvore.insert(235)
arvore.print()


