//Prática 4 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>

typedef int exponentType;
typedef double coeficientType;

class OrderedList{
    public:
        OrderedList();
        ~OrderedList();
        void insert(exponentType exponent, coeficientType coeficient);
        void replace(exponentType exponent, coeficientType coeficient);
        void remove(exponentType exponent);
        void overwriteByListSum(OrderedList &p, OrderedList &q);
        void overwriteByListProduct(OrderedList &p, OrderedList &q);
        void getNext(exponentType &exponent, coeficientType &coeficient);
        void getNextReset();
        void clear();
        void print();
        bool empty();
        bool full();
        int size();
        int degree();
        int search(exponentType exponent);
    private:
        struct listNode;
        typedef listNode (*listPointer);
        struct listNode{
            exponentType exponent; //KEY
            coeficientType coeficient;
            listPointer nextNode;
        };

        listPointer head, sentinel, pointer_next;
        int count;

        void insert_end(exponentType exponent, coeficientType coeficient);
};