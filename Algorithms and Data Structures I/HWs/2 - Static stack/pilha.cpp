//Prática 2 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include "pilha.h"

Stack::Stack(){
    top = 0;
};

Stack::~Stack(){
    std::cout << "\n" << "Stack closed." << "\n";
};

void Stack::push(entryType in){
    if(full()){
        std::cout << "Stack full!" << "\n";
        return;
    }

    top++;
    stackData[top] = in;
};

void Stack::pop(entryType &out){
    if(empty()){
        std::cout << "Stack empty!" << "\n";
        return;
    }

    out = stackData[top];
    top--;
};

bool Stack::empty(){
    return (top == 0);
};

bool Stack::full(){
    return (top == MAX);
};