#Lucas Miranda (12542838)
#O código criado está indicado com comentários.


class BSTnoh():
    info = ""
    esq = None
    dir = None
    def __init__(self, string = ''):
        self.info = string
        self.esq = None
        self.dir = None
    
    def getEsq(self):
        if self.esq != None:
            return self.esq
    def getDir(self):
        if self.dir != None:
            return self.dir
    def getInfo(self):
        return self.info
    
class BST():
    def __init__(self):
        self.root = None
        
    def insere(self, dado):
        self.root = self.__add__(self.root, dado)
        
    def __add__(self, noh, dado):
        if noh == None:
            return BSTnoh(dado)
        else:
            if dado < noh.getInfo():
                noh.esq = self.__add__(noh.esq, dado)
            elif dado > noh.getInfo():
                noh.dir = self.__add__(noh.dir, dado)
        return noh
        
    def inorder(self):
        """ Retorna lista de nos em ordem crescente """
        self.items = []
        self.__inorder__(self.root)
        return self.items
        
    def __inorder__(self, n):
        if(n == None):
            return ''
        else:
            self.__inorder__(n.getEsq())
            self.items.append(n.getInfo())
            self.__inorder__(n.getDir())
    def preorder(self):
        self.items = []
        self.__preorder__(self.root)
        return self.items
        
    def __preorder__(self, n):
        if(n == None):
            return ''
        else:
            self.items.append(n.getInfo())
            self.__preorder__(n.getEsq())
            self.__preorder__(n.getDir())
    def postorder(self):
        self.items = []
        self.__postorder__(self.root)
        return self.items
        
    def __postorder__(self, n):
        if(n == None):
            return ''
        else:
            self.__postorder__(n.getEsq())
            self.__postorder__(n.getDir())
            self.items.append(n.getInfo())     
        
        
    #Print da árvore completa deitada  ------------------------------------------
            
    def printa_arvore_bonita_deitada(self):
        self.__printa_arvore_bonita_deitada__(self.root, 0)
        
    def __printa_arvore_bonita_deitada__(self, n, level):
        if(n == None):
            return ''
        else:
            self.__printa_arvore_bonita_deitada__(n.getDir(), level+1)
            print("  "*level, n.getInfo())
            self.__printa_arvore_bonita_deitada__(n.getEsq(), level+1)
        
    # --------------------------------------------------------------------------- 
           
    #Print da árvore completa em pé --------------------------------------------- 
            
    def printa_arvore_bonita_em_pe(self):
        self.items = []
        self.niveis = []
        self.__printa_arvore_bonita_em_pe__(self.root, 0)
        
        spacing = len(str(max(self.items)))
        
        for i in range(max(self.niveis)+1):
            for j in range(self.niveis.count(i)):
                indexes = [k for k, x in enumerate(self.niveis) if x == i]
                
                if (j == 0):
                    number_of_spaces = indexes[0] * spacing
                else:
                    number_of_spaces = int((indexes[1] - indexes[0]) * (spacing-1) )
                
                print(number_of_spaces * " ", self.items[indexes[j]], end="")
            print()
           
    def __printa_arvore_bonita_em_pe__(self, n, level):
        if(n == None):
            return ''
        else:
            self.__printa_arvore_bonita_em_pe__(n.getEsq(), level+1)
            self.items.append(n.getInfo())
            self.niveis.append(level)
            self.__printa_arvore_bonita_em_pe__(n.getDir(), level+1)
            
    # ---------------------------------------------------------------------------  
              
                                          
t=BST()
t.insere(1)
t.insere(2)
t.insere(3)
t.insere(4)
t.insere(5)
t.insere(100)
t.insere(0)


#Chamando as funções ----------------------------------------------------------

print("\nArvore deitada:")
t.printa_arvore_bonita_deitada()

print("\nArvore em pé:")
t.printa_arvore_bonita_em_pe()

# -----------------------------------------------------------------------------


