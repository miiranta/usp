#include <stdio.h> 

int usaVetor(int *vetor);

void main(){
    int i = 0, vetor[1000];

    for(i = 0; i<1000; i++){
        vetor[i] = i;
    }

    usaVetor(vetor);

}

int usaVetor(int *vetor){

    int i = 0, buffer = 0, soma = 0;

    for(i = 0; i<1000; i = i + 2){
        buffer = vetor[i+1];
        vetor[i+1] = vetor[i];
        vetor[i] = buffer;
    }

    printf("Vetor:\n", vetor[i]);
    for(i = 0; i<1000; i++){
        printf(" %d", vetor[i]);
        soma = soma + vetor[i];
    }

    printf("\nSum:\n %d", soma);
}