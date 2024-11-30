//Q4 - LUCAS MIRANDA - Numero USP: 12542838

#include <stdio.h> 

void main(){
	
    int matr[][4] = {1,2,3,4,5,6,7,8,9,10,11,12};

    //(A) **matr é 1?
    //VERDADEIRO
    //**matr é o mesmo que matr[0][0]
    printf("A) %d\n", **matr );

    //(B) *(*(matr+1)+2) é 7?
    //VERDADEIRO
    //matr+1 faz o programa avançar uma linha (equivale a [1]), +2 faz o programa avançar 2 elementos (equivale a [1][2]), localização do numero 7.
    printf("B) %d\n", *(*(matr+1)+2) );

    //(C) *(matr[2]+3) é 12?
    //VERDADEIRO
    //matr[2] é a terceira linha (cada uma formada por 4), iniciada por 9. O +3 faz o programa andar 3 endereços, chegando até o 12.
    printf("C) %d\n", *(matr[2]+3) );


    //(D) (*(matr+2))[2] é 11?
    //VERDADEIRO
    //matr+2 é o mesmo que matr[2], ou seja, a terceira linha, iniciada em 9. O [2] faz o programa pegar o terceiro endereço dessa linha, ou seja, 11.
    printf("D) %d\n", (*(matr+2))[2] );

    //(E) *((*matr)+1) é 5?
    //FALSO
    //*matr é equivalente a matr[0], ou seja, a primeira linha. Ja (*matr)+1 é equivalente a matr[0][1], ou seja, o segundo elemento (2) e não o 5.
    printf("E) %d\n", *((*matr)+1) );
    
}

