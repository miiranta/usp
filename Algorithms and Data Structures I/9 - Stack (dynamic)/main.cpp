#include <iostream>
#include "stack.cpp"

using namespace std;

int main(){
    Stack stack;
    entryType in, out;

    in = 2;
    stack.push(in);
    in = 4;
    stack.push(in);
    in = 6;
    stack.push(in);

    cout << "Stack size: " << stack.size() << endl;
    stack.getTop(out);
    cout << "Stack top: " << out << endl;

    stack.pop(out);
    cout << out << endl;
    stack.pop(out);
    cout << out << endl;
    stack.pop(out);
    cout << out << endl;

    cout << "Stack size: " << stack.size() << endl;

    stack.pop(out);
}

