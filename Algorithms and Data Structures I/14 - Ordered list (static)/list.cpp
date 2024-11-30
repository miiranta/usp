#include <iostream>
#include "list.h"

OrderedList::OrderedList(){
    count = 0;
}

OrderedList::~OrderedList(){
    std::cout << "List closed!" << std::endl;
}

bool OrderedList::empty(){
    return (count == 0);
}

bool OrderedList::full(){
    return (count == MAX);
}

void OrderedList::insert(entryType in){  
    int i, j;

    if(full()){
        std::cout << "List is full." << std::endl;
        return;
    }

    //linear Search
    i = 1;
    while(i<=count && in>entry[i]){
        i++;
    }

    for(j=count; j>=i; j--){
        entry[j+1] = entry[j];
    }

    entry[i] = in;
    count++;
}

void OrderedList::remove(entryType in){  
    int i, p;

    if(empty()){
        std::cout << "List is empty." << std::endl;
        return;
    }

    p = search(in);

    if(p==0){
        std::cout << "Element not found." << std::endl;
        return;
    }

    for(i=p; i<count; i++){
        entry[i] = entry[i+1];
    }

    count--;
}

void OrderedList::replace(entryType in, int p){
    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    entry[p] = in;
}

void OrderedList::retrieve(entryType &out, int p){
    if(p<1 || p>count){
        std::cout << "Position out of range." << std::endl;
        return;
    }

    out = entry[p];
}

void OrderedList::clear(){
    count = 0;
}

int OrderedList::size(){
    return count;
}

int OrderedList::search(entryType in){
    //fast binary search
    int m, L = 1, R = count;

    while(L<R){
        m = (L+R)/2;

        if(entry[m] < in){
            L = m + 1;
        }else{
            R = m;
        }
    }

    return (entry[R] != in ? 0 : R);
}


