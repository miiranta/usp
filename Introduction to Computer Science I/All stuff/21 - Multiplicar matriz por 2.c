#include <stdio.h>
#include <string.h>

void main(){

//vars
int i, j;
int size = 3;
int matrix[100][100]; 

    printf("Insert matrix elements:\n");

    //Get elements
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){

        printf("i:%d j:%d = ",i ,j ) ;  
        scanf("%d", &matrix[i][j]);
        fflush(stdin);

        }
    }

    //Print elements
    for(i=0; i<size; i++){
        printf("\n");
        for(j=0; j<size; j++){
        printf("%d  ", matrix[i][j]);
        }
    }

    //Print elements without diagonal
    printf("\n");
    for(i=0; i<size; i++){
        printf("\n");
        for(j=0; j<size; j++){
        printf("%d  ", matrix[i][j]*2);
        }
    }



}