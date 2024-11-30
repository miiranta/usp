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
        print("Inserindo: ", key)
        self.insert_recursive(self.root, key)
           
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
        
        #Search and add key
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

        #Split
        if keys_num == 3:
            self.split(node)
            self.insert_recursive(self.root, key)
            
            return False
        
        #Search and add child   
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

    def search(self, key):
        return self.search_recursive(self.root, key)
    
    def search_recursive(self, node, key, auto_merge = False):
        
        if(node == None):
            return False
        
       
        
        keys_num = self.list_size(node.keys)
        
        for i in range(keys_num):
            current_key = node.keys[i]
            
            if(key == current_key):
                return node
            
            elif(key < current_key):
                return self.search_recursive(node.children[i], key, True)
        
        if(node.children[i+1] != None):
            return self.search_recursive(node.children[i+1], key, True)
        
        return False

    def delete(self, key):
        print("Deletando: ", key)
        
        node_to_delete = self.search_recursive(self.root, key, True)
        
        if(node_to_delete == False):
            print("Chave não encontrada.")
            return False
        
        return self.delete_recursive(node_to_delete, key)
        
    def delete_recursive(self, node, key):
        
        #Algumas vars
        key_index = node.keys.index(key)
           
        #CASO: Elemento é folha com pelo menos 2 chaves
        if(self.is_leaf(node) and self.list_size(node.keys) > 1):
            print("(-) Folha: tem pelo menos 2 chaves.")
            
            node.keys.remove(key)
            node.keys = self.list_sort(node.keys)
            
            return True
        
        #CASO: Elemento é nó interno
        if(self.is_leaf(node) == False):

            #Filho a esquerda tem pelo menos 2 chaves
            #Trocar chave por Sucessor e deletar
            if(node.children[key_index+1] != None and self.list_size(node.children[key_index+1].keys) > 1):
                print("(-) Interno: Filho esquerdo tem pelo menos 2 chaves.")
            
                sucessor = node.children[key_index+1].keys[0]
                node.keys[key_index] = sucessor
                node.children[key_index+1].keys[0] = None
                node.children[key_index+1].keys = self.list_sort(node.children[key_index+1].keys)

                return True
            
            #Filho a direita tem pelo menos 2 chaves
            #Trocar chave por Predecessor e deletar
            if(node.children[key_index] != None and self.list_size(node.children[key_index].keys) > 1):
                print("(-) Interno: Filho direito tem pelo menos 2 chaves.")

                last_key_index = self.list_size(node.children[key_index].keys) - 1
                predecessor = node.children[key_index].keys[last_key_index]
                node.keys[key_index] = predecessor
                node.children[key_index].keys[last_key_index] = None
                node.children[key_index].keys = self.list_sort(node.children[key_index].keys)
                
                return True
            
            #Os dois filhos tem 1 chave
            #Merge com o pai
            if(self.list_size(node.children[key_index].keys) == 1 and self.list_size(node.children[key_index+1].keys) == 1):
                print("(-) Interno: Filhos tem 1 chave.")
                
                #node = self.merge_parent(node.children[key_index], node.children[key_index+1])
                #return self.delete_recursive(node, key)
            
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
                        print("(-) Rotacionando", node.parent.children[i].keys[key_amount-1], "para a esquerda.")
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
                        print("(-) Rotacionando", node.parent.children[i].keys[0], "para a direita.")
                        self.rotate_right(node.parent.children[i])
        
        #Acabou?
        if(self.list_size(node.keys) >= 2):
            return True   
    
    def apply_merge_case(self, node):
        node_index = node.parent.children.index(node)
        parent = node.parent
        
        if((node_index+1 <= 4) and (parent.children[node_index+1] != None)):
            brother = parent.children[node_index+1]
        else:
            brother = parent.children[node_index-1]
        
        print("(-) Merge entre", node.keys[0], "e", brother.keys[0], "e uma chave do pai.")
        return self.merge_parent(node, brother)
    
    def merge_parent(self, node, brother):
        parent = node.parent
        
        node_index = node.parent.children.index(node)
        brother_index = node.parent.children.index(brother)
        parent_index = min([node_index, brother_index])
        
        node.keys.append(parent.keys[parent_index])
        node.keys.append(brother.keys[0])
        node.keys = self.list_sort(node.keys)
        
        parent.keys[parent_index] = None
        parent.keys = self.list_sort(parent.keys)
        
        parent.children[brother_index] = None
        
        self.organize_children(parent)
       
        #Fix: Parent pode ter ficado com 0 chaves
        # if(self.list_size(parent.keys) == 0):
        #     parent.parent.children.append(node)
        #     parent.parent.children.remove(parent)
        #     self.organize_children(parent.parent)
        # self.organize_parents(node)
        
        # #Fix: Parent pode ter ficado com 0 chaves
        # if(self.list_size(parent.keys) == 0):
        #     keys_num = self.list_size(node.keys)
            
        #     parent.keys.append(node.keys[keys_num-1])
        #     parent.keys = self.list_sort(parent.keys)
        #     node.keys[keys_num-1] = None
        #     node.keys = self.list_sort(node.keys)
            
        #     self.organize_children(parent)
        #     self.organize_parents(node)
            
        
        return node
       
    def merge_trivial(self, node):   
        
        #No máximo 1 chave no node.
        if(self.list_size(node.keys) > 1):
            return False
        
        #2 Filhos com 1 chave cada (Naturalmente, 2 filhos max).
        children_num = self.list_size(node.children)
        if((children_num != 2)):
            return False
        
        child1_num = self.list_size(node.children[0].keys)
        child2_num = self.list_size(node.children[1].keys)
        if(child1_num != 1 or child2_num != 1):
            return False
        
        #Tudo certo, faça o merge down.
        print("(-) Merge down.")
        node.keys.append(node.children[0].keys[0])
        node.keys.append(node.children[1].keys[0])
        node.keys = self.list_sort(node.keys)
        new_children_list = [node.children[0].children[0], node.children[0].children[1], node.children[1].children[0], node.children[1].children[1]]
        node.children = new_children_list
        node.children = self.organize_children(node)
        
        return True   
                                     
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
               
    def is_leaf(self, node):
        return node.children[0] == None and node.children[1] == None and node.children[2] == None and node.children[3] == None
    
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
     
    def organize_parents(self, node):
        children_num = self.list_size(node.children)
        
        for i in range(children_num):
            node.children[i].parent = node
        
        return node
      
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
       
                self.organize_children(node)
                node = self.root
                
                self.organize_parents(node1)
                self.organize_parents(node2)
                
                print("Splitting (root)")
                self.print()
                
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
                
                self.organize_children(node.parent)
                
                self.organize_parents(node1)
                self.organize_parents(node2)
                
                print("Splitting")
                self.print()
                
                return self.split(node.parent)
            
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

            #if(node.parent != None):
            #    print("       "*level,">Parent: ", node.parent.keys)

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


# ====================

arvore.delete(14)
arvore.print()

arvore.delete(449)
arvore.print()

arvore.delete(11)
arvore.print()

#arvore.delete(12)
#arvore.print()















