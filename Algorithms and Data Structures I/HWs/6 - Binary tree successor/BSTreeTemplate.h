//Nome:       Lucas Miranda Mendonça Rezende
//Número USP: 12542838

#include <iostream>
#include <iomanip>
#include <climits>
#include <sstream>
#include <cstring>
using namespace std;

#ifndef BSTREETEMPLATE_H
#define BSTREETEMPLATE_H

template< class TreeEntry >
class BinarySearchTree
{ public:
    BinarySearchTree();
    ~BinarySearchTree();
    bool empty();
    bool full();
    void clear();
    bool search(TreeEntry x);
    void insert(TreeEntry x);
    bool remove(TreeEntry x);
    void print();
    string toString();
    int nodes();
    int leaves();
    int height();
    TreeEntry minimum();
    TreeEntry maximum();
    TreeEntry successor(TreeEntry x);
    
  private:
    // definicao de estruturas
    struct TreeNode; 

    typedef TreeNode *TreePointer; 

    struct TreeNode
    { TreeEntry entry;             
      TreePointer leftNode, rightNode; 
    };

    // campos
    TreePointer root;
    
    // metodos
    void clear(TreePointer &t);
    bool iSearch(TreeEntry x);
    bool rSearch(TreeEntry x, TreePointer &t);
    bool remove(TreeEntry x, TreePointer &p);
    void removeMin(TreePointer &q,  TreePointer &r);
    void preOrder(TreePointer &t);
    void inOrder(TreePointer &t);
    void postOrder(TreePointer &t);
    void print(TreePointer &t, int s);
    string toString(TreePointer &t);
    int  nodes(TreePointer &t);
    int  leaves(TreePointer &t);
    int  height(TreePointer &t);
    TreeEntry minimum(TreePointer &t);
    TreeEntry maximum(TreePointer &t);
}; 

//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: Arvore binaria e criada e iniciada como vazia
template< class TreeEntry >
BinarySearchTree<TreeEntry>::BinarySearchTree()
{  
  root = NULL;
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: Arvore e destruida, liberando espaco ocupado pelo seus elementos 
template< class TreeEntry >
BinarySearchTree<TreeEntry>::~BinarySearchTree()
{  
  clear();
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna true se a arvore binaria esta vazia; false caso contrario
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::empty()
{
  return (root == NULL);
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna true se a arvore binaria esta cheia; false caso contrario
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::full()
{
  return false;
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: todos os elementos da arvore sao descartados e ela torna-se uma arvore vazia
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::clear()
{
  clear(root);
  root = NULL;
}
//------------------------------------------------------------
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::clear(TreePointer &t)
{
  if( t != NULL )
  { clear( t->leftNode );
    clear( t->rightNode );
    delete t;
  }
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: Retorna true se x foi encontrado na arvore; false caso contrario
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::search(TreeEntry x)
{ // basta escolher uma unica implementacao do metodo de busca
  // return iSearch(x);
  return rSearch(x, root);
}
//------------------------------------------------------------
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::iSearch(TreeEntry x)
{ TreePointer t=root;

  while (t != NULL && t->entry != x) 
    if(x < t->entry)
      t = t->leftNode; // procurar subarvore esquerda
    else
      t = t->rightNode; // procurar subarvore direita
 
  return (t != NULL);
}
//------------------------------------------------------------
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::rSearch(TreeEntry x, TreePointer &t)
{
  if(t == NULL)
    return false; // x nao encontrado
  
  if(x < t->entry)
      return rSearch(x,t->leftNode);
  else
    if(x > t->entry)
      return rSearch(x,t->rightNode);
   else     // x == t->entry
      return true;
}
//------------------------------------------------------------
// pre-condicao: Arvore nao esta cheia
// pos-condicao: O item x e inserido na arvore 
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::insert(TreeEntry x)
{ TreePointer p=NULL, q=root, r;

  while (q != NULL) 
  { p = q; 
    if(x < q->entry)
      q = q->leftNode;
    else
      q = q->rightNode;
  }
  
  r = new TreeNode;
  r->entry = x;
  r->leftNode = NULL;
  r->rightNode = NULL;

  if(p == NULL)
    root = r;  // arvore vazia
  else
    if(x < p->entry)
      p->leftNode = r;
    else
      p->rightNode = r;
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna true se o item x foi encontrado e, portanto, removido da arvore; caso contrario, retorna false
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::remove(TreeEntry x)
{
  return remove(x,root);
}
//------------------------------------------------------------
template< class TreeEntry >
bool BinarySearchTree<TreeEntry>::remove(TreeEntry x, TreePointer &p)
{ TreePointer q;
   
   if(p == NULL)
     return false; // elemento inexistente

   if(x < p->entry)
      return remove(x,p->leftNode);
   else
    if(x > p->entry)
       return remove(x,p->rightNode);
    else // remover p->
    { q = p;
      if(q->rightNode == NULL)
         p = q->leftNode;
      else
        if(q->leftNode == NULL)
          p = q->rightNode;
        else
          removeMin(q,q->rightNode);
      delete q;
      return true;
    }
}
//------------------------------------------------------------
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::removeMin(TreePointer &q,  TreePointer &r)
{
  if(r->leftNode != NULL)
    removeMin(q,r->leftNode);
  else
  { q->entry = r->entry;
    q = r;
    r = r->rightNode;
  }
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: imprime a arvore
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::print(TreePointer &t, int s)
{ int i;

  if(t != NULL) 
  { print(t->rightNode, s+3);
    for(i=1; i<= s; i++)
      cout << " ";
    cout << setw(6) << t->entry << endl;
    print(t->leftNode, s+3);
  }
}
//------------------------------------------------------------
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::print()
{
  print(root,0);     
  cout << endl << "Pre-ordem" << endl;
  preOrder(root);
  cout << endl << "Em-ordem" << endl;
  inOrder(root);
  cout << endl << "Pos-ordem" << endl;
  postOrder(root);
  cout << endl;
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna arvore como string
template< class TreeEntry >
string BinarySearchTree<TreeEntry>::toString()
{ 
  return toString( root );
}
//------------------------------------------------------------
template< class TreeEntry >
string BinarySearchTree<TreeEntry>::toString(TreePointer &t)
{ 
  if(t != NULL) 
    return toString(t->leftNode) + 
	       static_cast<ostringstream*>(&(ostringstream() << t->entry))->str() + " " +
		   toString(t->rightNode);
  else
    return "";  // subarvore vazia, nada a fazer
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: percorre a arvore em pre-ordem
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::preOrder(TreePointer &t)
{
  if(t != NULL)
  { cout << t->entry << ",";
    preOrder(t->leftNode);
    preOrder(t->rightNode);
  }
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: percorre a arvore em em-ordem
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::inOrder(TreePointer &t)
{
  if(t != NULL)
  { inOrder(t->leftNode);
    cout << t->entry << ",";
    inOrder(t->rightNode);
  }
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: percorre a arvore em pos-ordem
template< class TreeEntry >
void BinarySearchTree<TreeEntry>::postOrder(TreePointer &t)
{
  if(t != NULL)
  { postOrder(t->leftNode);
    postOrder(t->rightNode);
    cout << t->entry << ",";
  }
}
//------------------------------------------------------------
// pre-condicao:  nenhuma
// pos-condicao: retorna o numero de nos existentes na arvore
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::nodes()
{  return nodes(root);
}
//------------------------------------------------------------
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::nodes(TreePointer &t)
{ 
  if(t == NULL)
     return 0;
  else
    return 1 + nodes(t->leftNode) + nodes(t->rightNode);
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna o numero de folhas existentes na arvore
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::leaves()
{  return leaves(root);
}
//------------------------------------------------------------
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::leaves(TreePointer &t)
{  if(t == NULL)
     return 0;
   else
     if(t->leftNode == NULL && t->rightNode == NULL)
        return 1;
     else
        return leaves(t->leftNode) + leaves(t->rightNode);
}
//------------------------------------------------------------
// pre-condicao: nenhuma
// pos-condicao: retorna a altura da arvore
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::height()
{  return height(root);
}
//------------------------------------------------------------
template< class TreeEntry >
int BinarySearchTree<TreeEntry>::height(TreePointer &t)
{ if(t == NULL)
     return -1;
  else
  {   int L,R;
      L = height(t->leftNode);
      R = height(t->rightNode);
      if(L>R) return L+1; else return R+1;
  }
}
//------------------------------------------------------------
// pre-condicao: Arvore nao esta vazia
// pos-condicao: Retorna o valor minimo encontrado na arvore binaria de busca
template< class TreeEntry >
TreeEntry BinarySearchTree<TreeEntry>::minimum()
{  if( root == NULL )
   {  cout << "Arvore vazia" << endl;
      return INT_MIN; 
   }
   return minimum(root);
}
//------------------------------------------------------------
template< class TreeEntry >
TreeEntry BinarySearchTree<TreeEntry>::minimum(TreePointer &t)
{  if( t->leftNode == NULL ) 
      return t->entry;
   else
     return minimum(t->leftNode);
}
//------------------------------------------------------------
// pre-condicao: Arvore nao esta vazia
// pos-condicao: Retorna o valor maximo encontrado na arvore binaria de busca
template< class TreeEntry >
TreeEntry BinarySearchTree<TreeEntry>::maximum()
{  if( root == NULL )
   {  cout << "Arvore vazia" << endl;
      return INT_MAX; 
   }
   return maximum(root);
}
//------------------------------------------------------------
template< class TreeEntry >
TreeEntry BinarySearchTree<TreeEntry>::maximum(TreePointer &t)
{  if( t->rightNode == NULL ) 
      return t->entry;
   else
     return maximum(t->rightNode);
}

//------------------------------------------------------------
template< class TreeEntry >
TreeEntry BinarySearchTree<TreeEntry>::successor(TreeEntry x)
{ //O pior caso é:
  //O(N) ou O(H) se a arvore for desbalanceada (já que H pode ser igual a N no pior caso)
  //O(logN) ou O(H) se a arvore for balanceada (já que H é proporcional a LogN no pior caso)
  //
  //Onde N é o numero de nos da arvore
  //e H é a altura da arvore
  
  TreePointer t = root, saveSuc;

  //Procurando o elemento para achar o sucessor
  while (t != NULL && t->entry != x){
    if(x < t->entry){

      //Se for para a esquerda, o sucessor é o parente da nova posição
      saveSuc = t;
      
      t = t->leftNode; // procurar subarvore esquerda
    }else{
      t = t->rightNode; // procurar subarvore direita
    }
  }
    
  //Elemento/Sucessor existe?
  if(t == NULL || saveSuc == NULL){
    //cout << "Sucessor nao encontrado";
    return INT_MAX;
  }
  
  //Caso 1: Existe árvore direita não vazia
  if(t->rightNode != NULL){
    t = t->rightNode;

    while(t->leftNode != NULL){
      t = t->leftNode;
    }
    return t->entry;
  }

  //Caso 2: Subárvore direita vazia
  return saveSuc->entry;
}
//------------------------------------------------------------
#endif /* BSTREETEMPLATE_H */

