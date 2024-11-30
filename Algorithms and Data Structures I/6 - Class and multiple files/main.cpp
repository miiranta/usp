#include <iostream>
#include "myClassFunctions.cpp"

using namespace std;
 
int main(){

    //Create an instance of the class
    //Constructor will be called once when the object is declared
    Hello h;

    //Call the method
    h.sayHello();
    h.showConst();

    //Testing static
    Hello h1;
    Hello h2;

    //Non static var
    h1.var_a = 11;
    cout << h1.var_a << endl;
    cout << h2.var_a << endl;

    //Static var
    h1.var_b = 22;
    cout << h1.var_b << endl;
    cout << h2.var_b << endl;

}