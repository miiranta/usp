//Prática 2 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include <string>
#include "pilha.cpp"

using namespace std;

int main(){
    Stack stack;
    entryType out;
    string frase;
    int i;

    //Pegando a sentença do usuário
    cout << "Insira uma frase:" << endl; 
    getline(cin, frase);
    
    //Para cada char da sentença, use stack.push(), até no máximo a pilha estar cheia.
    i = 0;
    while(!stack.full() && frase[i]){
        stack.push(frase[i]);
        i++;
    }

    //Enquanto a pilha não estiver vazia, use stack.pop()
    while(!stack.empty()){
        stack.pop(out);
        cout << out;
    }

}

