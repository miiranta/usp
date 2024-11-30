//Prática 2 - Algoritmos e Estruturas de dados 1
//Lucas Miranda (12542838)

#include <iostream>

typedef char entryType;

class Stack{
    public:
        Stack();
        ~Stack();
        void push(entryType in);
        void pop(entryType &out);
        bool empty();
        bool full();
    private:
        static const int MAX = 500;
        int top;
        entryType stackData[MAX + 1];
};
