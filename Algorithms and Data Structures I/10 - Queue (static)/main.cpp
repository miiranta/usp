#include <iostream>
#include "queue.cpp"

using namespace std;

int main(){
    Queue q;
    entryType in, out;

    //Should print "Queue is empty!"
    q.serve(out);

    q.append(1);
    q.append(2);
    q.append(3);
    q.append(4);
    q.append(5);
    
    //Should print "Queue is full!"
    q.append(6);

    //Should print itens added.
    q.serve(out);
    cout << out << endl;
    q.serve(out);
    cout << out << endl;
    q.serve(out);
    cout << out << endl;
    q.serve(out);
    cout << out << endl;
    q.serve(out);
    cout << out << endl;

    //More tests
    q.append(6);
    q.append(7);
    q.append(8);
    q.append(9);
    q.append(10);

    q.serve(out);
    cout << out << endl;
    q.serve(out);
    cout << out << endl;

    cout << "Queue size: " << q.size() << endl;
    q.getFront(out);
    cout << "Front: " << out << endl;
    q.getRear(out);
    cout << "Rear: " << out << endl;

}

