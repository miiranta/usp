#include <stdio.h>

void main(){

for(;;){

    double res = 1, soma = 0; 
    int n = 0;
    printf("\nN: ");
    scanf("%d", &n);

        if(n != 0){
            for(int i = 1; i<=n; i++){

            res = 1/(float)i;
            soma = res + soma;

            }
        }
        else if(n = 0){res = 0;}

    printf("%f", soma);

}

}