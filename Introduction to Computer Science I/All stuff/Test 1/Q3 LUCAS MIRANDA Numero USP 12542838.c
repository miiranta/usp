#include <stdio.h>
#include <string.h>
#include <math.h>

void main(){

    float a=1, b=1, c=1;
    float delta = 0;
    float resposta = 0;

    printf("Insert a: ");
    scanf("%f", &a);

    printf("Insert b: ");
    scanf("%f", &b);

    printf("Insert c: ");
    scanf("%f", &c);

    printf("\na:%f b:%f c:%f\n", a,b,c);

    delta = (float)pow(b,2)-4*a*c;
    printf("Delta: %f\n", delta);

    if(delta > 0){
        resposta = ((-b) + sqrt(delta)) / (2*a);
        printf("There are 2 roots! X1: %f", resposta);

        resposta = ((-b) - sqrt(delta)) / (2*a);
        printf(" X2: %f", resposta);

    }

    if(delta == 0){
        resposta = (-(float)b)/(2*(float)a);
        printf("Theres only one root! X1: %f", resposta);

    }

    if(delta < 0){
    printf("No roots!");
    }

}