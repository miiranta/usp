#include <iostream>
#include "list.cpp"

using namespace std;

int main(){
    List q;
    entryType in, out;

    q.insert(1, 1);
    q.insert(2, 2);

    q.retrieve(out, 1);
    cout << out << endl;

    q.remove(out, 1);
    cout << out << endl;

}