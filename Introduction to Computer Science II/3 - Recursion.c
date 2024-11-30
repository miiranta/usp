#include <stdio.h>
int factorial(int n);

void main(){

    printf("%d", factorial(4));

}


//factiorial by recursion
int factorial(int n){

    //Recursion
    if(n > 0){
        return n*factorial(n-1);
    }

    //Base
    if(n == 0){
        return 1; 
    }
  

}

//There are 2 fibonacci examples in the "Challenges" folder