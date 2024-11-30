#include <iostream>
#include "list.h"

OrderedList::OrderedList(){
    sentinel = new listNode;
    head = sentinel;
    count = 0;
}

OrderedList::~OrderedList(){
    clear();
    delete sentinel;
}

bool OrderedList::empty(){
    return (head == sentinel);
}

bool OrderedList::full(){
    return false;
}

void OrderedList::setPosition(int p, listPointer &current){
    int i;

    //Test if the position is valid
    if(p<1 || p>count+1){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    current = head;
    for(i=2; i<=p; i++){
        current = current->nextNode;
    }
}

void OrderedList::insert(entryType in){  
    listPointer p, q;

    //linear search with sentinel
    sentinel->entry = in;
    p = head;
    while(p->entry < in){
        p = p->nextNode;
    }

    q = new listNode;
    if(q == NULL){
        std::cout << "List is full" << std::endl;
        return;
    }

    if(p == sentinel){
        p->nextNode = q;
        sentinel = q;
    }else{
        *q = *p;
        p->entry = in;
        p->nextNode = q;
    }

    count++;
}

void OrderedList::remove(entryType in){  
    listPointer p = NULL, q = head;

    //linear search with sentinel
    sentinel->entry = in;
    while(q->entry < in){
        p = q;
        q = q->nextNode;
    }

    //Find in?
    if(q->entry != in || q == sentinel){
        return;
    }

    //Removal place
    if(q == head){
        head = q->nextNode; //at the start
    }else{
        p->nextNode = q->nextNode; //others
    }

    delete q;
    count--;
}

void OrderedList::replace(entryType in, int p){
    listPointer current;

    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    setPosition(p, current);
    current->entry = in;
}

void OrderedList::retrieve(entryType &out, int p){
    listPointer current;

    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    setPosition(p, current);
    out = current->entry;
}

void OrderedList::searchInsert(entryType in){
    listPointer p, q;

    //linear search with sentinel
    sentinel->entry = in;
    p = head;
    while(p->entry < in){
        p = p->nextNode;
    }

    if(p != sentinel && p->entry == in){
        //FOUND (increase frequency)
        p->frequency++;

    }else{
        //NOT FOUND (insert)
        q = new listNode;

        if(q == NULL){
            std::cout << "List is full." << std::endl;
            return;
        }

        if(p == sentinel){
            p->nextNode = q;
            sentinel = q;
        }else{
            *q = *p;
            p->entry = in;
            p->nextNode = q;
        }

        p->frequency = 1;
        count++;
    }
}

void OrderedList::clear(){
    listPointer q;

    while(head != sentinel){
        q = head;
        head = head->nextNode;
        delete q;
    }

    count = 0;
}

int OrderedList::size(){
    return count;
}

int OrderedList::search(entryType in){
    //linear search with sentinel
    int pos = 1;
    listPointer q = head;

    sentinel->entry = in;
    while(q->entry < in){
        q = q->nextNode;
        pos++;
    }

    if(q->entry != in || q == sentinel){
        return 0;
    }else{
        return pos;
    }
}


