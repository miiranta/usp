//Prática 4 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include "list.h"

OrderedList::OrderedList(){
    sentinel = new listNode;
    head = sentinel;
    count = 0;
    pointer_next = head;
}

OrderedList::~OrderedList(){
    clear();
    delete sentinel;
}

void OrderedList::replace(exponentType exponent, coeficientType coeficient){  
    listPointer p, q;

    if(exponent < 0){
        std::cout << "Exponent must be positive." << std::endl;
        return;
    }

    //linear search with sentinel
    sentinel->exponent = exponent;
    sentinel->coeficient = coeficient;
    p = head;
    while(p->exponent > exponent){
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

    }else if(p->exponent == exponent){
        p->exponent = exponent;
        p->coeficient = coeficient;
        delete q;

    }else{
        *q = *p;
        p->exponent = exponent;
        p->coeficient = coeficient;
        p->nextNode = q;
    }

    count++;
}

void OrderedList::insert(exponentType exponent, coeficientType coeficient){  
    listPointer p, q;

    if(exponent < 0){
        std::cout << "Exponent must be positive." << std::endl;
        return;
    }

    //linear search with sentinel
    sentinel->exponent = exponent;
    sentinel->coeficient = coeficient;
    p = head;
    while(p->exponent > exponent){
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

    }else if(p->exponent == exponent){
        p->exponent = exponent;
        p->coeficient = coeficient + p->coeficient; //sum of coeficients!!
        delete q;

    }else{
        *q = *p;
        p->exponent = exponent;
        p->coeficient = coeficient;
        p->nextNode = q;
    }

    count++;
}

void OrderedList::insert_end(exponentType exponent, coeficientType coeficient){  
    listPointer p, q;

    if(exponent < 0){
        std::cout << "Exponent must be positive." << std::endl;
        return;
    }

    q = new listNode;
    if(q == NULL){
        std::cout << "List is full" << std::endl;
        return;
    }

    p = sentinel;
    p->nextNode = q;
    sentinel = q;

    p->exponent = exponent;
    p->coeficient = coeficient;

    count++;
}

void OrderedList::overwriteByListSum(OrderedList &p, OrderedList &q){
    exponentType expP, expQ;
    coeficientType coefP, coefQ;
    bool endP = false, endQ = false;

    if(p.empty() || q.empty()){
        std::cout << "One of the lists is empty." << std::endl;
        return;
    }

    clear();

    p.getNextReset();
    q.getNextReset();

    p.getNext(expP, coefP);
    q.getNext(expQ, coefQ);

    while(true){
        
        //Did someone reach the end?
        if(expP == -1){
            endP = true;
        }
        if(expQ == -1){
            endQ = true;
        }

        //Break?
        if(endQ && endP){
            return;
        }

        //Where to insert whats found?
        if(expP > expQ && !endP){
            insert_end(expP, coefP);
            p.getNext(expP, coefP);
        }
        if(expP == expQ && !endP && !endQ){
            insert_end(expP, coefP + coefQ);
            p.getNext(expP, coefP);
            q.getNext(expQ, coefQ);
        }
        if(expP < expQ && !endQ){
            insert_end(expQ, coefQ);
            q.getNext(expQ, coefQ);
        }
    }
}

void OrderedList::overwriteByListProduct(OrderedList &p, OrderedList &q){
    exponentType expP, expQ;
    coeficientType coefP, coefQ;

    if(p.empty() || q.empty()){
        std::cout << "One of the lists is empty." << std::endl;
        return;
    }

    clear();

    p.getNextReset();
    q.getNextReset();

    p.getNext(expP, coefP);
    q.getNext(expQ, coefQ);

    while(expP != -1){
        while(expQ != -1){
            insert(expP + expQ, coefP * coefQ);
            q.getNext(expQ, coefQ);
        }
        p.getNext(expP, coefP);
        q.getNext(expQ, coefQ);
    }
}

void OrderedList::remove(exponentType exponent){  
    listPointer p = NULL, q = head;

    if(exponent < 0){
        std::cout << "Exponent must be positive." << std::endl;
        return;
    }

    //linear search with sentinel
    sentinel->exponent = exponent;
    while(q->exponent > exponent){
        p = q;
        q = q->nextNode;
    }

    //Find in?
    if(q->exponent != exponent || q == sentinel){
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

void OrderedList::getNext(exponentType &exponent, coeficientType &coeficient){
    exponent = -1;
    coeficient = -1;
    
    if(empty()){
        std::cout << "List is empty." << std::endl;
        return;
    }

    if(pointer_next == NULL || pointer_next == sentinel){
        pointer_next = head;
        return;
    }

    exponent = pointer_next->exponent;
    coeficient = pointer_next->coeficient;

    pointer_next = pointer_next->nextNode;
}

void OrderedList::getNextReset(){
    pointer_next = head;
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

void OrderedList::print(){
    listPointer p = head;

    if(empty()){
        std::cout << "List is empty." << std::endl;
        return;
    }

    bool loop = true;
    while(loop){
        if(p->exponent != 0){
            std::cout << p->coeficient << "x^"<< p->exponent;
        }else{
            std::cout << p->coeficient;
        }
        
        p = p->nextNode;
        
        if(p != sentinel){
            std::cout << " + ";
        }else{
            loop = false;
        }
    }

    std::cout << "\n";
}

bool OrderedList::empty(){
    return (head == sentinel);
}

bool OrderedList::full(){
    return false;
}

int OrderedList::size(){
    return count;
}

int OrderedList::degree(){
    if(empty()){
        return 0;
    }
    return head->exponent;
}

int OrderedList::search(exponentType exponent){
    //linear search with sentinel
    int pos = 1;
    listPointer q = head;

    sentinel->exponent = exponent;
    while(q->exponent > exponent){
        q = q->nextNode;
        pos++;
    }

    if(q->exponent != exponent || q == sentinel){
        return 0;
    }else{
        return pos;
    }
}


