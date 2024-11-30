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
        void clear();
        bool empty();
        bool full();
        int size();
        int search(entryType in);
    private:
        static const int MAX = 100;
        int count;
        entryType entry[MAX+1];
};