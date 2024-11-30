#include <iostream>
using namespace std;

struct iceCream{
    bool isGood;
};

int main(){
    
    //Pointer of type iceCream
    iceCream *ice = new iceCream;

    //You can assign values as follows:
    ice->isGood = true;

    cout << ice->isGood;

    //It is similar to ice.isGood, we just use -> if ice is a pointer
    
}
