#include <stdio.h> 

float calculaN(int n);

void main(){
	
    int n = 10;
    float res = 0;

    res = calculaN(n);
    printf("Sum: %f", res);

}

float calculaN(int n){

    int i = 0;
    float inverso = 1, termo = 1, soma = 0;

    for(i = 1; i<=n; i++){

        termo = ((2*(float)i)/(2*(float)i-1))*inverso;
        inverso = -inverso;

        printf("Term %d: %f\n",i, termo);
        soma = soma + termo;
    }

    return soma;
}
