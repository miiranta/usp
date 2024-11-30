#include <iostream>
#include "queue.h"

Queue::Queue(){
    head = tail = NULL;
    count = 0;
}

Queue::~Queue(){
    clear();
    std::cout << "Queue closed!" << std::endl;
}

bool Queue::empty(){
    return (head == NULL);
}

bool Queue::full(){
    return false;
}

void Queue::append(entryType in){
    QueuePointer p = new QueueNode;

    if(p == NULL){
        std::cout << "Queue is full!" << std::endl;
        return;
    }

    p->entry = in;

    if(empty()){
        head = tail = p;
    }else{
        tail->nextNode = p;
        tail = p;
    }

    p->nextNode = NULL;
    count++;
}

void Queue::serve(entryType &out){
    QueuePointer p;

    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    out = head->entry;
    p = head;
    head = head->nextNode;
    delete p;

    if(head == NULL){
        tail = NULL;
    }
    
    count--;
}

void Queue::clear(){
    entryType x;

    while(!empty()){
        serve(x);
    }

    count = 0;
}

void Queue::getFront(entryType &out){
    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    out = head->entry;
}

void Queue::getRear(entryType &out){
    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    out = tail->entry;
}

int Queue::size(){
    return count;
}

