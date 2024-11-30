#include <iostream>
#include "queue.h"

Queue::Queue(){
    count = 0;
    head = 1;
    tail = MAX_SIZE;
}

Queue::~Queue(){
    std::cout << "Queue closed!" << std::endl;
}

bool Queue::empty(){
    return (count == 0);
}

bool Queue::full(){
    return (count == MAX_SIZE);
}

void Queue::append(entryType in){
    if(full()){
        std::cout << "Queue is full!" << std::endl;
        return;
    }

    count++;
    tail = (tail % MAX_SIZE) + 1;
    entry[tail] = in;
}

void Queue::serve(entryType &out){
    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    count--;
    out = entry[head];
    head = (head % MAX_SIZE) + 1;
}

void Queue::clear(){
    count = 0;
    head = 1;
    tail = MAX_SIZE;
}

void Queue::getFront(entryType &out){
    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    out = entry[head];
}

void Queue::getRear(entryType &out){
    if(empty()){
        std::cout << "Queue is empty!" << std::endl;
        return;
    }

    out = entry[tail];
}

int Queue::size(){
    return count;
}

