#include <stdio.h>
#include <stdlib.h>

void gg(int matrixA[4][6], int matrixB[6][4], int matrixC[4][4]){
    int i, j, k;
    int buffer;
    
    //Para cada linha de A
    for(i = 0; i<4; i++){
    //Para cada coluna de B
       for(j = 0; j<4; j++){
           buffer = 0;
           //Para cada elemento das fileiras
           for(k = 0; k<6; k++){
               buffer = buffer + matrixA[i][k] * matrixB[k][j];
           } 
           matrixC[i][j] = buffer;

       }
    }
}

main ()
{
    int matrixA[4][6] = {
    {1,2,3,4,5,6},
    {1,2,3,4,5,7},
    {7,8,9,1,2,3},
    {1,2,4,6,8,9}
    }; 

    int matrixB[6][4] = {
    {1,2,3,4},
    {5,6,7,8},
    {9,1,2,3},
    {4,5,6,7},
    {8,9,1,2},
    {3,4,5,6}
    }; 
    
    int matrixC[4][4];
    
    int i, j;

    gg(matrixA, matrixB, matrixC);
    
    for(i = 0; i<4; i++){
        printf("\n");
        for(j = 0; j<4; j++){
             printf("%d ", matrixC[i][j]);
        }
    }
}