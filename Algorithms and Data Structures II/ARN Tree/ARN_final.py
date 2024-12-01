#Lucas Miranda Mendonça Rezende
#12542838

class Node:
    
    #Define atributos de um nó
    def __init__(self, value):
        self.null = True 
        self.color = "B"
        self.key = value
        self.left = None
        self.right = None
        self.parent = None

class ARN: 
    
    def __init__(self):
        self.root = None

    #Inserção
    def insert(self, value):
        
        #Novo nó
        node = Node(value)
        node.null = False
        node.color = "R"
        self.add_nulls(node)
        
        #É raiz?
        if self.root == None:
            self.root = node
            node.color = "B"
        else:
            print("Inserindo:", node.key)
            self.insert_node(self.root, node)
            
    def insert_node(self, root, node):

        #Inserção na esquerda
        if node.key < root.key:
            if root.left.null == True:
                root.left = node
                node.parent = root
            else:
                self.insert_node(root.left, node)
        
        #Inserção na direita
        else:
            if root.right.null == True:
                root.right = node
                node.parent = root
            else:
                self.insert_node(root.right, node)
        
        #Balanceamento 
        self.insert_balance(node)
        
    def insert_balance(self, node):
        
        #Definindo pai, tio e avo
        parent = node.parent
        grandparent = None
        uncle = None
        
        if(parent != None):
            grandparent = parent.parent
        
            if grandparent != None:
                if grandparent.left == node.parent:
                    uncle = grandparent.right
                else:
                    uncle = grandparent.left
            
        if(parent != None and uncle != None):
        
            #Caso 1: Pai e Tio são vermelhos
            if(parent.color == "R" and uncle.color == "R"):
                print("(+) Caso 1")  
                parent.color = "B"
                uncle.color = "B"
                
                if(grandparent == self.root):
                    grandparent.color = "B"
                else:
                    grandparent.color = "R"
                    
                self.insert_balance(grandparent)
            
            #Caso 2: Pai é vermelho e tio é preto
            if(parent.color == "R" and uncle.color == "B"):
            
                    #Caso 2.1
                    if(parent.left == node and grandparent.left == parent):
                        print("(+) Caso 2.1")          
                        parent.color = "B"
                        grandparent.color = "R"
                        self.rotate_right(grandparent)
                    
                    #Caso 2.2
                    if(parent.right == node and grandparent.right == parent):
                        print("(+) Caso 2.2")                       
                        parent.color = "B"
                        grandparent.color = "R"
                        self.rotate_left(grandparent)
                    
                    #Caso 2.3
                    if(parent.right == node and grandparent.left == parent):
                        print("(+) Caso 2.3")
                        self.rotate_left(parent)
                        self.insert_balance(parent)
                    
                    #Caso 2.4
                    if(parent.right == node and grandparent.left == parent):
                        print("(+) Caso 2.4")
                        self.rotate_right(parent)
                        self.insert_balance(parent)
    
    #Adiciona nós nulos
    def add_nulls(self, node):
        if(node.left == None and node.null == False):
            node.left = Node(None)
            node.left.parent = node
        if(node.right == None and node.null == False):
            node.right = Node(None)
            node.right.parent = node

    #Rotates            
    def rotate_right(self, node):
        
        if(node == None):
            return
        
        #Caso root
        if(node == self.root):
            old_root = self.root
            
            self.root = node.left
            self.root.right.parent = old_root
            self.root.parent = None
            
            old_root.left = self.root.right
            old_root.parent = self.root
            
            self.root.right = old_root
        
        #Caso interno
        else:
            node.left.parent = node.parent
            
            if(node.parent.left == node):
                node.parent.left = node.left
            else:
                node.parent.right = node.left
                
            aux = node.left.right
            node.left.right = node
            node.parent = node.left
            node.left = aux
            aux.parent = node
                 
    def rotate_left(self, node):
           
        if(node == None):
            return
        
        #Caso root
        if(node == self.root):
            old_root = self.root
            
            self.root = node.right
            self.root.left.parent = old_root
            self.root.parent = None
            
            old_root.right = self.root.left
            old_root.parent = self.root
            
            self.root.left = old_root
        
        #Caso interno  
        else:
            node.right.parent = node.parent
            
            if(node.parent.left == node):
                node.parent.left = node.right
            else:
                node.parent.right = node.right
                
            aux = node.right.left
            node.right.left = node
            node.parent = node.right
            node.right = aux
            aux.parent = node
                    
    #Deleção
    def delete(self, value):
        
        #Pesquisa pelo nó
        node = self.search(value)
        
        #Se achar, deleta
        if(node != None):
            print("Apagando:", node.key)
            self.delete_node(node)
        else:
            print("Valor não encontrado")
    
    def delete_node(self, node): 
        self.delete_balance(node, node)
          
    def delete_balance(self, node, original):
         
        #Caso: Nó é raiz e é folha
        if(self.is_leaf(node) and node == self.root):
            print("(-) Raiz folha")
            self.root = None
            return
         
        #Caso: folhas rubras
        if(self.is_leaf(node) and node.color == "R"):
            print("(-) Folha rubra")
            self.remove_refs(original)
            return
        
        #Caso: 2 filhos
        if(node == original):
            if(node.left != None and node.right != None):
                    
                    predecessor = self.find_predecessor(node)
                    successor = self.find_successor(node)
                    
                    if(predecessor != None):
                        print("(-) 2 filhos")
                        node.key = predecessor.key
                        self.delete_balance(predecessor, predecessor)
                        return
                    
                    elif(successor != None):
                        print("(-) 2 filhos")
                        node.key = successor.key
                        self.delete_balance(successor, successor)
                        return
                    
        #Caso: nós negros
        x = node #Nó a ser deletado
        y = node.left #Nó a ser movido
        
        if(node.right != None and node.right.null == False):
            y = node.right
        elif(node.left != None and node.left.null == False):
            y = node.left
            
        #Define irmão e pai do nó
        parent = x.parent
        if x == self.root:
            brother = None
        elif x.parent.left == x:
            brother = x.parent.right
        else:
            brother = x.parent.left

        if y != None: 
            
            if(y != self.root and brother != None):
                
                #Caso 1: irmão rubro
                if(brother.color == "R"):
                    print("(-) Caso 1")
                    parent.color = "R"
                    brother.color = "B"
                    
                    if(brother == parent.right):
                        self.rotate_left(parent)
                    else:
                        self.rotate_right(parent)
                    
                    self.delete_balance(node, original)
                    return
                
                if(brother.left != None and brother.right != None):
                
                    #Caso 2: Irmão é negro e ambos os filhos são negros
                    if(brother.color == "B" and brother.right.color == "B" and brother.left.color == "B"):
                        print("(-) Caso 2")
                        brother.color = "R"
                        
                        if(parent.color == "R"):
                            parent.color = "B"
                            self.remove_refs(original)
                        else:
                            self.remove_refs(original)
                            self.delete_balance(parent, original)
                            
                        return
                    
                    #Irmão na direita
                    if(brother == parent.right):
                    
                        #Caso 3.1: Irmão é negro e filho direito do irmão é negro
                        if(brother.color == "B" and brother.right.color == "B"):
                            print("(-) Caso 3.1")
                            
                            brother.left.color = "B"
                            brother.color = "R"
                            self.rotate_right(brother)
                            self.delete_balance(node, original)
                            return
                            
                        #Caso 4.1: Irmão é negro e filho direito do irmão é rubro
                        if(brother.color == "B" and brother.right.color == "R"):
                            print("(-) Caso 4.1")
                            
                            brother.color = parent.color
                            parent.color = "B"
                            brother.right.color = "B"
                            
                            self.rotate_left(parent)
                            self.print()

                            if(self.root.color == "R"):
                                self.root.color = "B"

                            self.remove_refs(original)
                            return
                    
                    #Irmão na esquerda
                    else:
                        
                        #Caso 3.2: Irmão é negro e filho esquerdo do irmão é negro
                        if(brother.color == "B" and brother.left.color == "B"):
                            print("(-) Caso 3.2")
                            
                            brother.right.color = "B"
                            brother.color = "R"
                            self.rotate_left(brother)
                            self.delete_balance(node, original)
                            return
                            
                        #Caso 4.2: Irmão é negro e filho esquerdo do irmão é rubro
                        if(brother.color == "B" and brother.left.color == "R"):
                            print("(-) Caso 4.2")
                            
                            brother.color = parent.color
                            parent.color = "B"
                            brother.left.color = "B"
                            
                            self.rotate_right(parent)
                            
                            if(self.root.color == "R"):
                                self.root.color = "B"
                                
                            self.remove_refs(original)
                            return
    
    #Procura um nó (pela chave)
    def search(self, value):
        return self.search_node(self.root, value)
    
    def search_node(self, root, value): 
        if root == None or root.key == None:
            return None
        elif root.key == value:
            return root
        elif value < root.key:
            return self.search_node(root.left, value)
        else:
            return self.search_node(root.right, value)
    
    #Remove as referencias para o nó
    def remove_refs(self, node):
        if(node.parent != None):
            if(node.parent.left == node):
                node.parent.left = None
            elif(node.parent.right == node):
                node.parent.right = None
            
            self.add_nulls(node.parent)
                   
    #Achar nó sucessor/predecessor
    def find_successor(self, node):
        if(node.right != None and node.right.null == False):
            return self.find_min(node.right)
    
    def find_predecessor(self, node):
        if(node.left != None and node.left.null == False):
            return self.find_max(node.left)
    
    #Achar mínimo/máximo
    def find_min(self, node):
        if(node.left != None and node.left.null == False):
            return self.find_min(node.left)
        return node
    
    def find_max(self, node):
        if(node.right != None and node.right.null == False):
            return self.find_max(node.right)
        return node
           
    #É folha?
    def is_leaf(self, node):
        if(node.left != None and node.right != None):
            return node.left.null == True and node.right.null == True
        return False
    
    #Imprime a árvore
    def print(self):
        print("R-------------------")
        self.print_recursive(self.root, 0)
        print("L-------------------")
        print("\n")
    
    def print_recursive(self, node, level):
        
        if(node == None):
            return
        
        if node.right != None:
            self.print_recursive(node.right, level+1)
        
        if(node.null == False):   
            print("       "*level, node.key, node.color)
        else:
            print("       "*level, "NULL", node.color)
            
        if node.left != None:
            self.print_recursive(node.left, level+1)
    
    
#Cria a árvore
arvore = ARN()

#Insere os valores
arvore.insert(10)
arvore.print()

arvore.insert(5)
arvore.print()

arvore.insert(4)
arvore.print()

arvore.insert(20)
arvore.print()

arvore.insert(40)
arvore.print()

arvore.insert(1)
arvore.print()

arvore.insert(2)
arvore.print()

arvore.insert(60)
arvore.print()

arvore.insert(30)
arvore.print()

arvore.insert(25)
arvore.print()

arvore.insert(3)
arvore.print()

arvore.insert(7)
arvore.print()

arvore.insert(11)
arvore.print()

#Remove os valores
arvore.delete(25)
arvore.print()

arvore.delete(20)
arvore.print()

arvore.delete(30)
arvore.print()

arvore.delete(1)
arvore.print()

arvore.delete(3)
arvore.print()

arvore.delete(5)
arvore.print()

arvore.delete(40)
arvore.print()

arvore.delete(60)
arvore.print()

arvore.delete(4)
arvore.print()

arvore.delete(7)
arvore.print()

arvore.delete(11)
arvore.print()

arvore.delete(10)
arvore.print()

arvore.delete(2)
arvore.print()







