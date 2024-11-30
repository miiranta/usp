//Nome:       Lucas Miranda Mendonça Rezende
//Número USP: 12542838

#include <iostream>
#include <math.h>
#include "avl.h"

using namespace std;

template< class TreeEntry >
AVL<TreeEntry>::AVL()
{  
    root = NULL;
    wordCount = 0;
    mfwCount = 0;
}

template< class TreeEntry >
AVL<TreeEntry>::~AVL()
{  
    clear(root);
}

template< class TreeEntry >
bool AVL<TreeEntry>::empty()
{
  return (root == NULL);
}

template< class TreeEntry >
bool AVL<TreeEntry>::full()
{
  return false;
}

template< class TreeEntry >
void AVL<TreeEntry>::clear(TreePointer &t)
{
  if( t != NULL )
  { 
    clear( t->leftNode );
    clear( t->rightNode );
    delete t;
  }
}

template< class TreeEntry >
int AVL<TreeEntry>::nodes(TreePointer &t)
{ 
  if(t == NULL)
     return 0;
  else
    return 1 + nodes(t->leftNode) + nodes(t->rightNode);
}

template< class TreeEntry >
int AVL<TreeEntry>::leaves(TreePointer &t)
{  
    if(t == NULL)
        return 0;
    else
        if(t->leftNode == NULL && t->rightNode == NULL)
            return 1;
        else
            return leaves(t->leftNode) + leaves(t->rightNode);
}

template< class TreeEntry >
int AVL<TreeEntry>::height(TreePointer &t)
{ 
    if(t == NULL)
        return -1;
    else
    {   int L,R;
        L = height(t->leftNode);
        R = height(t->rightNode);
        if(L>R) return L+1; else return R+1;
    }
}

template< class TreeEntry >
int AVL<TreeEntry>::words()
{ 
  return wordCount;
}

template< class TreeEntry >
int AVL<TreeEntry>::smallestHeight(TreePointer &t)
{ 
    float base = 2, no_elements = (float) nodes(t);
    int hmin = ceil( (log(no_elements + 1) / log(base)) - 1 ); 

    return hmin;
}

template< class TreeEntry >
void AVL<TreeEntry>::searchInsert(TreeEntry x) // método público
{ 
    bool h = false;
    searchInsert(x, root, h);
}

template< class TreeEntry >
void AVL<TreeEntry>::searchInsert(TreeEntry x, TreePointer &pA, bool &h)
{ 
    TreePointer pB, pC;
    if(pA == NULL) // inserir
    { 
        pA = new TreeNode;
        h = true;
        pA->entry = x;
        pA->frequency = 1;
        pA->leftNode = pA->rightNode = NULL;
        pA->bal = 0;
        wordCount++;

        //word frequency list
        for(int i = 1; i < mfwMAX; i++)
        {
            if(mfw[i].entry == "")
            {
                mfw[i].entry = pA->entry;
                mfw[i].frequency = pA->frequency;
                mfwCount++;
                break;
            }
        }

    }else if(x < pA->entry){ 
        searchInsert(x, pA->leftNode, h);

        if(h) // subárvore esquerda cresceu
        { 
            switch (pA->bal){ 
                case -1: pA->bal = 0; h = false; break;
                case 0: pA->bal = +1; break;
                case +1: pB = pA->leftNode;
                if(pB->bal == +1) // rotação LL
                { 
                    pA->leftNode = pB->rightNode; pB->rightNode = pA;
                    pA->bal = 0; pA = pB;
                }else // rotação LR
                { 
                    pC = pB->rightNode; pB->rightNode = pC->leftNode;
                    pC->leftNode = pB; pA->leftNode = pC->rightNode;
                    pC->rightNode = pA;
                    if(pC->bal == +1) pA->bal = -1; else pA->bal = 0;
                    if(pC->bal == -1) pB->bal = +1; else pB->bal = 0;
                    pA = pC;
                }
                pA->bal = 0; h = false;
            } // switch
        }
    }else if(x > pA->entry){ 
        searchInsert(x, pA->rightNode, h);

        if(h) // subárvore direita cresceu
        { 
            switch (pA->bal){ 
                case +1: pA->bal = 0; h = false; break;
                case 0: pA->bal = -1; break;
                case -1: pB = pA->rightNode;
                if(pB->bal == -1) // rotação RR
                { 
                    pA->rightNode = pB->leftNode; pB->leftNode = pA;
                    pA->bal = 0; pA = pB;
                }else // rotação RL
                { 
                    pC = pB->leftNode; pB->leftNode = pC->rightNode;
                    pC->rightNode = pB; pA->rightNode = pC->leftNode;
                    pC->leftNode = pA;
                    if(pC->bal == -1) pA->bal = +1; else pA->bal = 0;
                    if(pC->bal == +1) pB->bal = -1; else pB->bal = 0;
                    pA = pC;
                }
                pA->bal = 0; h = false;
            } // switch
        }
    }else{ // elemento encontrado
        pA->frequency++;
        wordCount++;

        //word frequency list
        mostFrequentWords aux;
        for(int i = 1; i < mfwCount; i++)
        {
            if(mfw[i].entry == pA->entry)
            {
                mfw[i].entry = pA->entry;
                mfw[i].frequency = pA->frequency;
                break;
            }
        }

    } 
    
}

template< class TreeEntry >
void AVL<TreeEntry>::mostFrequent(TreePointer &t)
{ 
    int printHowMany = 20;

    sortList(mfw, mfwCount);

    for (int i = mfwCount; i > (mfwCount - printHowMany) && (i >= 1); i--)
    {
        cout << " -" << AVL::mfw[i].entry << " " << AVL::mfw[i].frequency << endl;
    }
}

template< class TreeEntry >
void AVL<TreeEntry>::write()
{ 
    cout << "Total de palavras: " << words() << endl;
    cout << "Total de palavras distintas: " << nodes(root) << endl;
    cout << "Altura da arvore minima: " << smallestHeight(root) << endl;

    cout << "Altura AVL: " << height(root) << endl;
    cout << "Total de folhas AVL: " << leaves(root) << endl;

    cout << "Palavras mais frequentes: " << endl;
    mostFrequent(root);

    return;
}


//QUICK SORT - Lista de palavras mais frequentes
template< class TreeEntry >
void AVL<TreeEntry>::sortList(mostFrequentWords *a, int N){
    qsort(a, 1, N);
}
template< class TreeEntry >
void AVL<TreeEntry>::qsort(mostFrequentWords *a, int L, int R){
    int fl = floor((L+R)/2);
    mostFrequentWords w;
    int i=L, j=R, x=a[fl].frequency;

    do{
        while(a[i].frequency<x){
            i = i + 1;
        }
        while(x<a[j].frequency){
            j = j - 1;
        }
        if(i<=j){
            w = a[i];
            a[i] = a[j];
            a[j] = w;
            i = i + 1;
            j = j -1;
        }

    }while(i<=j);
    if(L<j){qsort(a, L, j);}
    if(i<R){qsort(a, i, R);}
}
