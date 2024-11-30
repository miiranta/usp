#include <iostream>

typedef int entryType;

class Stack{
    public:
        Stack();
        ~Stack();
        void push(entryType in);
        void pop(entryType &out);
        void clear();
        void getTop(entryType &out);
        bool empty();
        bool full();
        int size();
    private:
        //Linear nodes
        struct stackNode;
        typedef stackNode (*StackPointer);
        struct stackNode{
            entryType entry;
            StackPointer nextNode;
        };

        StackPointer top;
        int count;
};
