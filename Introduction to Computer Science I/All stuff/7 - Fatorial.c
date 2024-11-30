#include <stdio.h>

void main(){

for(;;){

    double res = 1; 
    int n = 0;
    printf("\nValor fatorial de: ");
    scanf("%d", &n);

        if(n != 0){
            for(int i = 1; i<=n; i++){

            res = res*i;

            }
        }
        else if(n == 0){res = 1;}

    printf("%f", res);

}

}