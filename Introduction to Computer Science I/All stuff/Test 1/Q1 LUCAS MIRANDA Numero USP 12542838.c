#include <stdio.h>
#include <string.h>
#include <math.h>

void main(){

int x=0, fatorial=1, i=1;
float fator=0, resposta=0, power=0;

printf("Insert number bigger than 2: ");
scanf("%d", &x);
printf("i: %d x: %d", i, x);

for(i = 2; i<=x; i++){

    fatorial = fatorial*i;

    power = 0;
    power = (float)pow(x, i);

    fator = fator + power/(float)fatorial;

    printf("\nx:%d power:%f fatorial:%d fator:%f", i ,power, fatorial, fator);

}

resposta = 1 + (float)x + fator;
printf("\n\n Response: %f", resposta);

}