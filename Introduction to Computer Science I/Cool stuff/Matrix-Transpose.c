#include <stdio.h>
#include <string.h>

void main(){

//vars
int n = 3; //nXn
int i, j;
float matrix[100][100]; 

    printf("Matrix size (NxN): ") ;  
    scanf("%d", &n);

    printf("Insert elements(float):\n");

    for(i=0; i<n; i++){
        for(j=0; j<n; j++){

        printf("i:%d j:%d = ",i ,j ) ;  
        scanf("%f", &matrix[i][j]);
        fflush(stdin);

        }
    }

    printf("\nMatrix:\n");
    for(i=0; i<n; i++){
        printf("\n");
        for(j=0; j<n; j++){
        printf("%f  ", matrix[i][j]);
        }
    }

    printf("\n\nTransposed Matrix:\n");
    for(i=0; i<n; i++){
        printf("\n");
        for(j=0; j<n; j++){
            printf("%f  ", matrix[j][i]);
        }
    }



}