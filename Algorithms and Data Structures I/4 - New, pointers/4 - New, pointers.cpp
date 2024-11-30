#include <iostream>
using namespace std;

int main(){

    //Dleclares, but DOES NOT allocate space;
    int *pointer;

    cout << "Address (without new): " << pointer << endl;

    //Gets a new space inside the memory the size of an int.
    pointer = new int;

    cout << "Address (with new): " << pointer << endl;

    //Putting some value
    *pointer = 10;
    cout << *pointer << "\n\n";

    //Using new for arrays
    int *array;
    
    int size = 1;
    cout << "Set size: " << endl;
    cin >> size;

    array = new int[size];

    for(int i=0; i < size; i++){
        array[i] = i;
    } 

    for(int i=0; i < size; i++){
        cout << array[i] << endl;
    } 


}

