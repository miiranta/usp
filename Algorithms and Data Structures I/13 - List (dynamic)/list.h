#include <iostream>

typedef int entryType;

class List{
    public:
        List();
        ~List();
        void insert(entryType in, int p);
        void remove(entryType &out, int p);
        void replace(entryType in, int p);
        void retrieve(entryType &out, int p);
        //void minimum(entryType &out, int &p);
        //void maximum(entryType &out, int &p);
        //void copy(List &l);
        //void reverse(); //For that list
        //void reverse(List &l); //For new list
        //void sort();
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
        };

        listPointer head;
        int count;

        void setPosition(int pos, listPointer &current);
};