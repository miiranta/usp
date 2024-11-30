//Nome:       Lucas Miranda Mendonça Rezende
//Número USP: 12542838

#include <iostream>

template< class TreeEntry >
class AVL
{ public:
    AVL();
    ~AVL();
    void searchInsert(TreeEntry x);
    void write();
    
  private:

    // definicao de estruturas
    struct TreeNode; 
    typedef TreeNode *TreePointer; 

    struct TreeNode
    { 
        TreeEntry entry;             
        TreePointer leftNode, rightNode; 
        int bal;
        int frequency;
    };

    // campos
    TreePointer root;
    int wordCount;

    // word frequency list
    int mfwCount;
    static const int mfwMAX = 50000;
    struct mostFrequentWords{
        TreeEntry entry;
        int frequency;
    };
    mostFrequentWords mfw[mfwMAX + 1];
  
    // metodos
    bool empty();
    bool full();
    void clear(TreePointer &t);
    void sortList(mostFrequentWords *a, int N);
    void qsort(mostFrequentWords *a, int L, int R);

    void searchInsert(TreeEntry x, TreePointer &pA, bool &h);
    void mostFrequent(TreePointer &t);
    int smallestHeight(TreePointer &t); //Árvore mínima?
    int nodes(TreePointer &t); //Palavras distintas
    int leaves(TreePointer &t); //Folhas
    int height(TreePointer &t); //Altura
    int words(); //Palavras totais

}; 