//Nome:       Lucas Miranda Mendonça Rezende
//Número USP: 12542838

#include <iostream>
#include <fstream>
#include "avl.cpp"
#include "abb.cpp"

using namespace std;

int main(){

    string diretorio = "";

    //Código dado para realizar leitura de arquivo
       
        //Deixe pelo menos um comentado por vez!
        //======================
        ABB<string> bst;
        //AVL<string> avl;
        //======================
        
        string word;

        //Escolha o arquivo!
        //======================
        ifstream theInput(diretorio + "mam.txt");
        //ifstream theInput(diretorio + "exame_10000.txt");
        //ifstream theInput(diretorio + "Bible-KJV.txt");
        //======================

        // ler cada palavra do arquivo e inserir
        while( theInput >> word )
        {   
            //Deixe pelo menos um comentado por vez!
            //======================
            bst.searchInsert(word);
            //avl.searchInsert(word);
            //======================
        }

        theInput.close();

        //Deixe pelo menos um comentado por vez!
        //======================
        cout << "\nABB ==========================================" << endl;
        bst.write();
        cout << "\nAVL ==========================================" << endl;
        //avl.write();
        //======================

        return 0;
    //----------------------------------

    

}