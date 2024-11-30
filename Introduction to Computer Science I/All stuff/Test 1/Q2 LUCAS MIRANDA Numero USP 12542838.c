#include <stdio.h>
#include <string.h>
#include <math.h>

void main(){

int i, n, fatorial=1, soma=0;

do{
printf("Insert a number x: ");
fflush(stdin);
scanf("%d", &n);

if(n<=30){

    fatorial = 1;

    for(i = 1; i<=n; i++){
        fatorial = fatorial*i;
    }

printf("\nFatorial: %d\n", fatorial);

}

if(n>30){
    soma = 0;

    for(i = -5; i<=n; i++){

        soma = soma + i;

    }

    printf("\nSoma: %d\n", soma);
}

}while(n != -1);


}