#include <iostream>

typedef int entryType;

class Stack{
    public:
        Stack();
        ~Stack();
        void push(entryType in);
        void pop(entryType &out);
        bool empty();
        bool full();

        int size();
        void getTop(entryType &out);
        void clear();

    private:
        static const int MAX = 500;
        int top;
        entryType stackData[MAX + 1];
};
