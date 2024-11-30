#Classe nó
class Node:
    
    #Define atributos do nó
    def __init__(self):
        self.parent = None
        self.children = [None, None, None, None] #Maximo: 4
        self.keys = [None, None, None] #Maximo: 3

class Tree234:
    
    def __init__(self):
        self.root = None
    
    #Inserção  
    def insert(self, key):
        print("Inserindo:", key)
        self.insert_recursive(self.root, key)
           
    def insert_recursive(self, node, key):
        
        if(node != None):
            keys_num = self.list_size(node.keys)
        
        #Se o nó for nulo, cria um novo nó
        else:
            node = Node()
            node.keys.append(key)
            node.keys = self.list_sort(node.keys)
            
            if(self.root == None):
                self.root = node
            
            return node
        
        #Se for folha, adiciona a chave no nó
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

        #Faça os split, se necessário
        if keys_num == 3:
            self.split(node)
            self.insert_recursive(self.root, key)
            
            return False
        
        #Se não for folha, procure o child a ser inserido 
        if (keys_num > 0 and key < node.keys[0]):
            res = self.insert_recursive(node.children[0], key)
            
            if(res == False):
                return False
            
            node.children[0] = res
            
            if(node.children[0] != None):
                node.children[0].parent = node
 
        elif (node.keys[1] == None) or (keys_num > 1 and key < node.keys[1]):
            res = self.insert_recursive(node.children[1], key)        
               
            if(res == False):
                return False    
                   
            node.children[1] = res
            
            if(node.children[1] != None):
                node.children[1].parent = node
        
        elif (node.keys[2] == None) or (keys_num > 2 and key < node.keys[2]):
            res = self.insert_recursive(node.children[2], key)
            
            if(res == False):
                return False
            
            node.children[2] = res
            
            if(node.children[2] != None):
                node.children[2].parent = node
        
        elif (node.keys[3] == None) or (keys_num > 3 and key < node.keys[3]):
            res = self.insert_recursive(node.children[3], key)
            
            if(res == False):
                return False
            
            node.children[3] = res
            
            if(node.children[3] != None):
                node.children[3].parent = node
        
        return node

    #Split: Quebra o nó em 2
    def split(self, node):
            
        if node != None:
            keys_num = self.list_size(node.keys)
                
        if keys_num == 3:
            
            #Caso root
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
       
                self.organize_children(node)
                node = self.root
                
                self.organize_parents(node1)
                self.organize_parents(node2)
                
                print("(+) Splitting root")
                
                return node
            
            #Caso interno
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
                
                self.organize_children(node.parent)
                
                self.organize_parents(node1)
                self.organize_parents(node2)
                
                print("(+) Splitting")
                
                return self.split(node.parent)
            
        else:
            return node
    
    #Deleção
    def delete(self, key):
        print("Deletando:", key)
        
        #Procura a chave
        node_to_delete = self.search_recursive(self.root, key, True)
        
        if(node_to_delete == False):
            print("Chave não encontrada.")
            return False
        
        #Se existir, deleta
        return self.delete_recursive(node_to_delete, key)
        
    def delete_recursive(self, node, key):
        
        key_index = node.keys.index(key)
           
        #CASO: Elemento é folha com pelo menos 2 chaves
        if(self.is_leaf(node) and self.list_size(node.keys) > 1):
            print("(-) Folha: tem pelo menos 2 chaves")
            
            node.keys.remove(key)
            node.keys = self.list_sort(node.keys)
            
            return True
        
        #CASO: Elemento é nó interno
        if(self.is_leaf(node) == False):

            #Filho a esquerda tem pelo menos 2 chaves
            #Trocar chave por Sucessor e deletar
            if(node.children[key_index+1] != None and self.list_size(node.children[key_index+1].keys) > 1):
                print("(-) Interno: Filho esquerdo tem pelo menos 2 chaves")
                print("(-) Copiando e deletando sucessor:")
            
                successor = self.get_successor(key)
                if(self.delete(successor) == False):
                    return False
                
                node.keys[key_index] = successor
                node.keys = self.list_sort(node.keys)
                
                return True
            
            #Filho a direita tem pelo menos 2 chaves
            #Trocar chave por Predecessor e deletar
            if(node.children[key_index] != None and self.list_size(node.children[key_index].keys) > 1):
                print("(-) Interno: Filho direito tem pelo menos 2 chaves")
                print("(-) Copiando e deletando predecessor:")

                predecessor = self.get_predecessor(key)
                if(self.delete(predecessor) == False):
                    return False
                
                node.keys[key_index] = predecessor
                node.keys = self.list_sort(node.keys)
                
                return True
            
            #Os dois filhos tem 1 chave
            #Merge com o no (o no tem pelo menos 2 chaves)
            if(self.list_size(node.children[key_index].keys) == 1 and self.list_size(node.children[key_index+1].keys) == 1):
                if(self.list_size(node.keys) > 1):  
                    print("(-) Interno: Filhos tem 1 chave e o no tem pelo menos 2 chaves")
                    
                    node = self.merge_parent(node.children[key_index], node.children[key_index+1])
                    return self.delete_recursive(node, key)
                
        #CASO: Não é nó interno (Outros casos)
        if(self.is_leaf(node)):
            parent = node.parent
            brothers = []
            
            if(parent != None):   
                brothers = node.parent.children.copy()
                brothers.remove(node)
   
            brothers_num = self.list_size(brothers)
            keys_num = self.list_size(node.keys)

            #Nó tem 1 chave e um irmão tem pelo menos 2 chaves
            #Rotacione o irmão para o pai e o pai para o elemento a ser deletado
            valid = False
            if(brothers_num >= 1 and keys_num == 1):
                for i in range(brothers_num):
                    if(self.list_size(brothers[i].keys) > 1):
                        valid = True
                        break
                    
            if(valid):
                self.apply_rotation_case(node)
                return self.delete_recursive(node, key)
                
            #Nó tem 1 chave e nenhum irmão tem pelo menos 2 chaves
            #Merge entre o nó, o irmão e uma chave do pai (o pai tem pelo menos 2 chaves, garantidas pelo merge down)
            valid = True
            parent_keys_num = self.list_size(parent.keys)
            if(brothers_num >= 1 and keys_num == 1 and parent_keys_num >= 2):
                for i in range(brothers_num):
                    if(self.list_size(brothers[i].keys) != 1):
                        valid = False
                        break
            else:
                valid = False
            
            if(valid):
                node = self.apply_merge_case(node)
                return self.delete_recursive(node, key)
    
        #CASO: Raiz com 1 chave e nenhum filho (extra ;))
        if(node == self.root and self.list_size(node.keys) == 1):
            if(self.list_size(node.children) == 0):
                print("(-) Raiz com 1 chave e 0 filhos")
                self.root = None
                return True
    
    #Procura um nó (pela chave)
    def search(self, key):
        return self.search_recursive(self.root, key)
    
    def search_recursive(self, node, key, merge_down = True):
         
        if(node == None):
            return False
        
        #Merge down automático (Obrigatório para a deleção)
        if(merge_down):
            if(self.merge_down(node) != False):
                node = self.root
                
        keys_num = self.list_size(node.keys)
        i = 0 
        for i in range(keys_num):
            current_key = node.keys[i]
            
            if(key == current_key):
                return node
            
            elif(key < current_key):
                return self.search_recursive(node.children[i], key, True)
        
        if(node.children[i+1] != None):
            return self.search_recursive(node.children[i+1], key, True)
        
        return False
    
    #Procura o predecessor/sucessor de um nó
    def get_predecessor(self, key):
        node_key = self.search_recursive(self.root, key, True)
        
        if(node_key == False):
            return False
        
        key_index = node_key.keys.index(key)
        
        return self.get_max(node_key.children[key_index])

    def get_successor(self, key):
        node_key = self.search_recursive(self.root, key, True)
        
        if(node_key == False):
            return False
        
        key_index = node_key.keys.index(key)
        
        return self.get_min(node_key.children[key_index+1])

    #Procura o menor/menor valor de um nó
    def get_max(self, node):
        keys_num = self.list_size(node.keys)
        
        if(self.is_leaf(node)):
            return node.keys[keys_num-1]
        
        return self.get_max(node.children[keys_num])
        
    def get_min(self, node):      
        if(self.is_leaf(node)):
            return node.keys[0]
        
        return self.get_min(node.children[0])

    #Casos de rotação e merge (deleção)
    def apply_rotation_case(self, node):
        node_index = node.parent.children.index(node)
        all_indexes = list(range(self.list_size(node.parent.children)))
        
        #Tentando rotacionar para a esquerda
        for i in all_indexes:
        
            #O nó está antes do nó a ser deletado?
            if(i < node_index):
                
                #O nó tem pelo menos 2 chaves?
                key_amount = self.list_size(node.parent.children[i].keys)
                if(key_amount >= 2):
                    
                    #O próximo nó existe e tem 1 chave?
                    if(node.parent.children[i+1] != None and self.list_size(node.parent.children[i+1].keys) == 1):
            
                        #Tudo certo, rotacione
                        print("(-) Rotacionando", node.parent.children[i].keys[key_amount-1], "para a esquerda")
                        self.rotate_left(node.parent.children[i])

        #Acabou?
        if(self.list_size(node.keys) >= 2):
            return True
  
        #Tentando rotacionar para a direita
        all_indexes.reverse()
        for i in all_indexes:
            
            #O nó está depois do nó a ser deletado?
            if(i > node_index):
                
                #O nó tem pelo menos 2 chaves?
                key_amount = self.list_size(node.parent.children[i].keys)
                if(key_amount >= 2):
                    
                    #O próximo nó existe e tem 1 chave?
                    if(node.parent.children[i-1] != None and self.list_size(node.parent.children[i-1].keys) == 1):
                        
                        #Tudo certo, rotacione
                        print("(-) Rotacionando", node.parent.children[i].keys[0], "para a direita")
                        self.rotate_right(node.parent.children[i])
        
        #Acabou?
        if(self.list_size(node.keys) >= 2):
            return True   
    
    def apply_merge_case(self, node):  
        node_index = node.parent.children.index(node)
        parent = node.parent
        
        #Define o irmão
        if((node_index+1 <= 4) and (parent.children[node_index+1] != None)):
            brother = parent.children[node_index+1]
        else:
            brother = parent.children[node_index-1]
        
        #Merge
        print("(-) Merge entre", node.keys[0], "e", brother.keys[0], "e uma chave do pai")
        return self.merge_parent(node, brother)
    
    #Merge: Junta dois nós (+ chave do pai)
    def merge_parent(self, node, brother):
        parent = node.parent
        
        node_index = node.parent.children.index(node)
        brother_index = node.parent.children.index(brother)
        parent_index = min([node_index, brother_index])
        
        node.keys.append(parent.keys[parent_index])
        node.keys.append(brother.keys[0])
        node.keys = self.list_sort(node.keys)
        
        node.children.append(brother.children[0])
        node.children.append(brother.children[1])
        
        parent.keys[parent_index] = None
        parent.keys = self.list_sort(parent.keys)
        
        parent.children[brother_index] = None
        
        self.organize_children(parent)
        self.organize_children(node) 
       
        #Se o parent ficou com 0 chaves
        if(self.list_size(parent.keys) == 0):
            
            #Caso não root
            if(parent != self.root and parent.parent != None):
                print(self.root.keys, parent.parent.keys)
                parent.parent.children.append(node)
                parent.parent.children.remove(parent)
                self.organize_children(parent.parent)
            
            #Caso Root
            else:
                self.root = node
                self.organize_children(self.root)
        
        self.organize_parents(node)

        return node
       
    #Merge automático da busca
    def merge_down(self, node):         
        
        #Filhos tem 1 chave
        for key in node.keys:
            if(key != None ):
                key_index = node.keys.index(key)
                
                if(node.children[key_index+1] != None):
                    child1_keys_num = self.list_size(node.children[key_index].keys)
                    child2_keys_num = self.list_size(node.children[key_index+1].keys)
                    
                    if(child1_keys_num == 1 and child2_keys_num == 1):
                        print("(-) Merge down em", node.keys[key_index])
                        return self.merge_parent(node.children[key_index], node.children[key_index+1])

        return False   
    
    #Rotação                       
    def rotate_left(self, node):
        parent = node.parent
        node_index = parent.children.index(node)
        brother_index = node_index + 1
        parent_index = node_index
        
        #Irmão existe?
        if(parent.children[brother_index] == None):
            return False
        brother = parent.children[brother_index]
        
        #Rotacione
        brother.keys.append(parent.keys[parent_index])
        brother.keys = self.list_sort(brother.keys)
        
        parent.keys.append(node.keys[self.list_size(node.keys)-1])
        parent.keys[parent_index] = None
        parent.keys = self.list_sort(parent.keys)
        
        node.keys[self.list_size(node.keys)-1] = None
        node.keys = self.list_sort(node.keys)
     
        return True
    
    def rotate_right(self, node):
        parent = node.parent
        node_index = parent.children.index(node)
        brother_index = node_index - 1
        parent_index = brother_index
        
        #Irmão existe?
        if(parent.children[brother_index] == None):
            return False
        brother = parent.children[brother_index]
        
        #Rotacione
        brother.keys.append(parent.keys[parent_index])
        brother.keys = self.list_sort(brother.keys)
        
        parent.keys.append(node.keys[0])
        parent.keys[parent_index] = None
        parent.keys = self.list_sort(parent.keys)
        
        node.keys[0] = None
        node.keys = self.list_sort(node.keys)
     
        return True
       
    #É folha?          
    def is_leaf(self, node):
        return node.children[0] == None and node.children[1] == None and node.children[2] == None and node.children[3] == None
    
    #Organiza os filhos de um nó (conforme as chaves)  
    def organize_children(self, node):
        sorted_children = self.list_sort_children(node.children)

        node.keys = self.list_sort(node.keys)
        node.children = [None, None, None, None]
        
        last = 0
        for i in range(len(sorted_children)):
            for j in range(len(node.keys)):
                
                if(node.keys[j] != None):
                    if(sorted_children[i] != None):
                        
                        if(sorted_children[i].keys[0] < node.keys[j]):
                            node.children[j] = sorted_children[i]
                            sorted_children[i] = None
                            last = last + 1
                            
        for i in range(len(sorted_children)):
            if(sorted_children[i] != None):
                node.children[last] = sorted_children[i]
                sorted_children[i] = None

        return node.children              
    
    #Organiza os pais de um nó
    def organize_parents(self, node):
        children_num = self.list_size(node.children)
        
        for i in range(children_num):
            node.children[i].parent = node
        
        return node
    
    #Retorna o tamanho de uma lista
    def list_size(self, list):
        return sum(x is not None for x in list)
    
    #Ordena uma lista
    def list_sort(self, list):
        filtered_list = filter(None, list)
        sorted_list = sorted(filtered_list)
        
        for i in range(3 - len(sorted_list)):
            sorted_list.append(None)
        
        return sorted_list
    
    #Ordena os filhos de um nó
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
    
    #Imprime a árvore 
    def print(self):
        print("R-------------------")
        self.print_recursive(self.root, 0)
        print("L-------------------")
        print("\n")
        
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

            #Mostra o parent do node!!
            #if(node.parent != None):
            #    print("       "*level,">Parent: ", node.parent.keys)

            print("       "*level,"--")


#Criando a árvore
arvore = Tree234()

#Insere os valores
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

arvore.insert(15)
arvore.print()

arvore.insert(158)
arvore.print()

arvore.insert(17)
arvore.print()

arvore.insert(1)
arvore.print()

arvore.insert(355)
arvore.print()

arvore.insert(449)
arvore.print()


#Remove os valores
arvore.delete(13)
arvore.print()

arvore.delete(14)
arvore.print()

arvore.delete(7)
arvore.print()

arvore.delete(11)
arvore.print()

arvore.delete(10)
arvore.print()

arvore.delete(3)
arvore.print()

arvore.delete(12)
arvore.print()

arvore.delete(4)
arvore.print()

arvore.delete(449)
arvore.print()

arvore.delete(355)
arvore.print()

arvore.delete(2)
arvore.print()

arvore.delete(1)
arvore.print()

arvore.delete(15)
arvore.print()

arvore.delete(6)
arvore.print()

arvore.delete(158)
arvore.print()

arvore.delete(17)
arvore.print()










