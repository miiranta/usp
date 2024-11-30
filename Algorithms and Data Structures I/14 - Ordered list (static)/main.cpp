#include <iostream>
#include "list.cpp"

using namespace std;

int main(){
    OrderedList q;
    entryType in, out;

    q.insert(2);
    q.insert(6);
    q.insert(8);
    q.insert(4);
    q.insert(5);
    q.insert(1);
    q.insert(7);
    q.insert(3);

    q.retrieve(out, 1);
    cout << out << endl;

    q.retrieve(out, 2);
    cout << out << endl;

    q.remove(2);

    q.retrieve(out, 2);
    cout << out << endl;


    

}