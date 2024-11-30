//Prática 3 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>
#include "queue.h"
using namespace std;

Queue::Queue(){
    head = tail = NULL;
    count = 0;
}

Queue::~Queue(){
    clear();
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
        cout << "Fila cheia." << endl;
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
        cout << "Fila vazia." << endl;
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
        cout << "Fila vazia." << endl;
        return;
    }

    out = head->entry;
}

void Queue::getRear(entryType &out){
    if(empty()){
        cout << "Fila vazia." << endl;
        return;
    }

    out = tail->entry;
}

int Queue::size(){
    return count;
}

void Queue::print(){
    QueuePointer p = head;
    int i = 0;

    if(empty()){
        cout << "Nao ha pessoas registradas nesse horario!" << endl;
        return;
    }

    while(p != NULL){
        i++;
        cout << i << " - " << p->entry.nome << endl;
        cout << "    Endereco:  " << p->entry.endereco << endl;
        cout << "    CPF:       " << p->entry.cpf << endl;
        cout << "    Idade:     " << p->entry.idade << "\n\n";

        p = p->nextNode;
    }

    delete p;
}


