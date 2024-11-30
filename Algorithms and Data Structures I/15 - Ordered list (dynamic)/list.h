#include <iostream>

typedef int entryType;

class OrderedList{
    public:
        OrderedList();
        ~OrderedList();
        void insert(entryType in);
        void remove(entryType in);
        void replace(entryType in, int p);      //from from 13 - List (dynamic) May be ineficient
        void retrieve(entryType &out, int p);   //from from 13 - List (dynamic) May be ineficient
        void searchInsert(entryType in);
        void clear();
        bool empty();
        bool full();
        int size();
        int search(entryType in);
    private:
        struct listNode;
        typedef listNode (*listPointer);
        struct listNode{
            entryType entry;
            listPointer nextNode;
            int frequency;
        };

        listPointer head, sentinel;
        int count;

        void setPosition(int p, listPointer &current); //from from 13 - List (dynamic) May be ineficient
};