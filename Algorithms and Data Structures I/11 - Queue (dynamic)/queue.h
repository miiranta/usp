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
        //Linear nodes
        struct QueueNode;
        typedef QueueNode (*QueuePointer);
        struct QueueNode{
            entryType entry;
            QueuePointer nextNode;
        };

        QueuePointer head, tail;
        int count;
};
