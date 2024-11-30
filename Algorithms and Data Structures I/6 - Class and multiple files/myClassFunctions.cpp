#include <iostream>
#include "myClass.h"

using namespace std;

//Costructor
Hello::Hello(){
    cout << "Hello!" << endl;
}

//Decosntructor
Hello::~Hello(){
    cout << "Bye!" << endl;
}

//public method
void Hello::sayHello(){
    cout << "Nice to meet you." << endl;
}

void Hello::showConst(){
    cout << var_c << endl;
}

