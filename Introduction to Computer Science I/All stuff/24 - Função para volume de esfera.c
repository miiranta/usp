#include <stdio.h>
#include <math.h>

float getVolume(float r) {

    return (4.0/3.0)*3.1415*pow(r,3);

}

void main(){

//vars
float radius = 0;

    //Get elements
    printf("Insert radius:\n");
    scanf("%f", &radius);
    fflush(stdin);

    printf("%f", getVolume(radius));

}

