#include <iostream>
#include "list.h"

List::List(){
    head = NULL;
    count = 0;
}

List::~List(){
    clear();
    std::cout << "List closed!" << std::endl;
}

bool List::empty(){
    return (head == NULL);
}

bool List::full(){
    return false;
}

void List::setPosition(int p, listPointer &current){
    int i;

    current = head;
    for(i=2; i<=p; i++){
        current = current->nextNode;
    }
}

void List::insert(entryType in, int p){  
    listPointer newNode, current;

    if(p<1 || p>count+1){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    newNode = new listNode;
    newNode->entry = in;

    if(p==1){
        newNode->nextNode = head;
        head = newNode;
    }else{
        setPosition(p-1, current);
        newNode->nextNode = current->nextNode;
        current->nextNode = newNode;
    }

    count++;
}

void List::remove(entryType &out, int p){  
    listPointer node, current;

    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    if(p==1){
        node = head;
        head = head->nextNode;
    }else{
        setPosition(p-1, current);
        node = current->nextNode;
        current->nextNode = node->nextNode;
    }

    out = node->entry;
    delete node;

    count--;
}

void List::replace(entryType in, int p){
    listPointer current;

    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    setPosition(p, current);
    current->entry = in;
}

void List::retrieve(entryType &out, int p){
    listPointer current;

    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    setPosition(p, current);
    out = current->entry;
}

void List::clear(){
    listPointer node;

    while(head != NULL){
        node = head;
        head = head->nextNode;
        delete node;
    }

    count = 0;
}

int List::size(){
    return count;
}

int List::search(entryType in){
    //linear search
    int p = 1;
    listPointer current = head;

    while(current != NULL && current->entry != in){
        current = current->nextNode;
        p++;
    }

    return (current == NULL ? 0 : p);
}



