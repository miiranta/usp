#include <iostream>
#include "nodes.h"

using namespace std;

main(){
    //Using directly
        StackNode s;
        s.entry = 1;
        s.nextNode = NULL;

        cout << s.entry << endl;

    //Using pointers
        StackPointer p;
        p = new StackNode;
        (*p).entry = 2;
        (*p).nextNode = NULL;
        //Or using p->entry = 1; and p->nextNode = NULL;

        cout << p->entry << endl;

        delete p; //Avoid memory leak

    
}