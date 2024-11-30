#include <iostream>

typedef int entryType;

class Queue{
    public:
        Queue();
        ~Queue();
        void append(entryType in);
        void serve(entryType &out);
        void getFront(entryType &out);
        void getRear(entryType &out);
        void clear(); 
        bool empty();
        bool full();
        int size();
    private:
        //Circular array implementation!
        static const int MAX_SIZE = 5;
        int head;
        int tail;
        int count;
        entryType entry[MAX_SIZE + 1];
};
