#include <stdio.h>

void main(){

    for(;;){

        int n = 1, count = 1;
        float soma = 0;

        printf("\nInsira um valor: ");
        scanf("%d", &n);

        for(int i = 1; i<=n*2; i++){

            if(i%2 == 1){

                float fac = 1;

                for(int j = 1; j<=i; j++){

                    fac = fac*j;

                }

                soma = soma + (float)count/fac;
                

                printf("\n%d / %f", count, fac);

                count++;

            }
        }

        printf("\nSoma: %f\n", soma);

    }
}