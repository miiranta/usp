//Nome:       Lucas Miranda Mendonça Rezende
//Número USP: 12542838

#include <iostream>
#include <math.h>
#include "abb.h"

using namespace std;

template< class TreeEntry >
ABB<TreeEntry>::ABB()
{  
    root = NULL;
    wordCount = 0;
    mfwCount = 0;
}

template< class TreeEntry >
ABB<TreeEntry>::~ABB()
{  
    clear(root);
}

template< class TreeEntry >
bool ABB<TreeEntry>::empty()
{
  return (root == NULL);
}

template< class TreeEntry >
bool ABB<TreeEntry>::full()
{
  return false;
}

template< class TreeEntry >
void ABB<TreeEntry>::clear(TreePointer &t)
{
  if( t != NULL )
  { 
    clear( t->leftNode );
    clear( t->rightNode );
    delete t;
  }
}

template< class TreeEntry >
int ABB<TreeEntry>::nodes(TreePointer &t)
{ 
  if(t == NULL)
     return 0;
  else
    return 1 + nodes(t->leftNode) + nodes(t->rightNode);
}

template< class TreeEntry >
int ABB<TreeEntry>::leaves(TreePointer &t)
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
int ABB<TreeEntry>::height(TreePointer &t)
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
int ABB<TreeEntry>::words()
{ 
  return wordCount;
}

template< class TreeEntry >
int ABB<TreeEntry>::smallestHeight(TreePointer &t)
{ 
    float base = 2, no_elements = (float) nodes(t);
    int hmin = ceil( (log(no_elements + 1) / log(base)) - 1 ); 

    return hmin;
}

template< class TreeEntry >
void ABB<TreeEntry>::searchInsert(TreeEntry x)
{ 
    searchInsert(x, root);
}

template< class TreeEntry >
void ABB<TreeEntry>::searchInsert(TreeEntry x, TreePointer &t)
{ 
    if( t == NULL )
    { 
        t = new TreeNode;
        t->entry = x;
        t->frequency = 1;
        t->leftNode = t->rightNode = NULL;
        wordCount++;

        //word frequency list
        for(int i = 1; i < mfwMAX; i++)
        {
            if(mfw[i].entry == "")
            {
                mfw[i].entry = t->entry;
                mfw[i].frequency = t->frequency;
                mfwCount++;
                break;
            }
        }

    }else if( x < t->entry ){
        searchInsert( x, t->leftNode );
    }
    else if( x > t->entry ){
        searchInsert( x, t->rightNode );
    }
    else{
        t->frequency++;
        wordCount++;

        //word frequency list
        mostFrequentWords aux;
        for(int i = 1; i < mfwCount; i++)
        {
            if(mfw[i].entry == t->entry)
            {
                mfw[i].entry = t->entry;
                mfw[i].frequency = t->frequency;
                break;
            }
        }

    }
}

template< class TreeEntry >
void ABB<TreeEntry>::mostFrequent(TreePointer &t)
{ 
    int printHowMany = 20;

    sortList(mfw, mfwCount);

    for (int i = mfwCount; i > (mfwCount - printHowMany) && (i >= 1); i--)
    {
        cout << " -" << ABB::mfw[i].entry << " " << ABB::mfw[i].frequency << endl;
    }
}

template< class TreeEntry >
void ABB<TreeEntry>::write()
{ 
    cout << "Total de palavras: " << words() << endl;
    cout << "Total de palavras distintas: " << nodes(root) << endl;
    cout << "Altura da arvore minima: " << smallestHeight(root) << endl;

    cout << "Altura ABB: " << height(root) << endl;
    cout << "Total de folhas ABB: " << leaves(root) << endl;

    cout << "Palavras mais frequentes: " << endl;
    mostFrequent(root);

    return;
}


//QUICK SORT - Lista de palavras mais frequentes
template< class TreeEntry >
void ABB<TreeEntry>::sortList(mostFrequentWords *a, int N){
    qsort(a, 1, N);
}
template< class TreeEntry >
void ABB<TreeEntry>::qsort(mostFrequentWords *a, int L, int R){
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
