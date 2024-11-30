#include <stdio.h>

void main(){

for(;;){

    float res = 1, soma = 0, fac = 1; 
    int n = 0;

    printf("\nCalcular ate: ");
    scanf("%d", &n);


        if(n != 0){

            for(int i = 1; i<=n; i++){

            float fi = (float)i;

            res = res*i;

            fac = fi/res;
            soma = fac + soma;

            printf("%f/%f \n", fi, res);
            }

        }
        else if(n = 0){res = 0;}

    printf("%f", soma);

}

}

