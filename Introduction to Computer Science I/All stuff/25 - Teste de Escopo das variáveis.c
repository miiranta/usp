#include <stdio.h>

//Global
int a = 3;

float getVolume() {

    printf("%d\n", a);

}

void main(){

    //Local
    int a = 4;

    getVolume();
    printf("%d\n", a);
   

}