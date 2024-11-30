#include <iostream>
#include "list.h"

List::List(){
    count = 0;
}

List::~List(){
    std::cout << "List closed!" << std::endl;
}

bool List::empty(){
    return (count == 0);
}

bool List::full(){
    return (count == MAX);
}

void List::insert(entryType in, int p){
    if(full()){
        std::cout << "List is full!" << std::endl;
        return;
    }

    if(p<1 || p>count+1){
        std::cout << "Invalid position!" << std::endl;
        return;
    }
    
    for(int i=count; i>=p; i--){
        entry[i+1] = entry[i];
    }

    entry[p] = in;
    count++;
}

void List::remove(entryType &out, int p){
    if(empty()){
        std::cout << "List is empty!" << std::endl;
        return;
    }

    if(p<1 || p>count){
        std::cout << "Invalid position!" << std::endl;
        return;
    }

    out = entry[p];
    for(int i=p; i<count; i++){
        entry[i] = entry[i+1];
    }

    count--;
}

void List::replace(entryType in, int p){
    if(p<1 || p>count){
        std::cout << "Invalid position!" << std::endl;
        return;
    }

    entry[p] = in;
}

void List::retrieve(entryType &out, int p){
    if(p<1 || p>count){
        std::cout << "Invalid position!" << std::endl;
        return;
    }

    out = entry[p];
}

void List::clear(){
    count = 0;
}

int List::size(){
    return count;
}

int List::search(entryType in){
    //linear search
    int p = 1;
    
    while(p<=count && entry[p] != in){
        p++;
    }

    return (p > count ? 0 : p);
}
