//Avoid multiple includes
#ifndef HELLO_DEF
#define HELLO_DEF

//Code
#include <iostream>

class Hello{
    public:
        Hello();
        ~Hello();
        void sayHello();
        void showConst();
        int var_a = 10;
        static int var_b;
    private:
        static const int var_c = 30;
};


//Yes, this is necessary in order to access static variables
int Hello::var_b = 20;

//Avoid multiple includes
#endif