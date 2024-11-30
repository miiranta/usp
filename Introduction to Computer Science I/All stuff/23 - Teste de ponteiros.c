#include <stdio.h>

void main ()
{

    //vars
    int *p; //Aqui
    int i, j;
    int size = 4;
    int matrix[size][size]; 
    int count = 0;

    p = matrix[0]; //Aqui

    //Get elements
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){

        *p = count; //Aqui
        p++;
        count++;

        }
    }

    //Print elements
    for(i=0; i<size; i++){
        printf("\n");
        for(j=0; j<size; j++){
        printf("%d  ", matrix[i][j]);
        }
    }


}
