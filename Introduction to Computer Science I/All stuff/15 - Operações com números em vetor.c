#include <stdio.h>
#include <math.h>

void main(){

    //Valores
    int vetor[10] = {2, 4, 20, 8, 7, 10, 5};

    int max = 0, min = 99999, soma = 0, j;
    float media = 0;


    //Soma
    for(j = 0; vetor[j]; j++){

        soma = soma + vetor[j];
        
    }

    printf("\nSoma: %d", soma);


    //Media
    media = (float)soma / (float)j ;

    printf("\nMedia: %d/%d = %f", soma, j, media);


    //Maior Valor
    for(j = 0; vetor[j]; j++){

        if(vetor[j] > max){ max = vetor[j]; }
        
    }

    printf("\nMaior Valor: %d", max);


    //Menor Valor
    for(j = 0; vetor[j]; j++){

        if(vetor[j] < min){ min = vetor[j]; }
        
    }

    printf("\nMenor Valor: %d", min);


}