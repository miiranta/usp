#include <stdio.h>
#include <string.h>

void main(){

int n = 3; //nXn
int i, j;
float matrix[100][100]; 

    printf("Insira elements (float):\n");

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

    printf("\n\nMatriz Transpose:\n");
    for(i=0; i<n; i++){
        printf("\n");
        for(j=0; j<n; j++){
            printf("%f  ", matrix[j][i]);
        }
    }

}