#include <iostream>
#include "stack.h"

Stack::Stack(){
    top = 0;
};

Stack::~Stack(){
    std::cout << "\n" << "Stack closed." << "\n";
};

void Stack::push(entryType in){
    if(full()){
        std::cout << "Stack is full!" << "\n";
        return;
    }

    top++;
    stackData[top] = in;
};

void Stack::pop(entryType &out){
    if(empty()){
        std::cout << "Stack is empty!" << "\n";
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

void Stack::clear(){
    top = 0;
};

void Stack::getTop(entryType &out){
    if(empty()){
        std::cout << "Stack is empty!" << "\n";
        return;
    }

    out = stackData[top];
};

int Stack::size(){
    return top;
};