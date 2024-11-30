#include <stdio.h>

int *gg(int matrixA[4][6], int matrixB[6][4]){

    static int result[4][4];
    int buffer = 0;

    //Para cada linha de A
    for(int i = 0; i<4; i++){
    //Para cada coluna de B
    for(int j = 0; j<4; j++){

        buffer = 0;

        //Para cada elemento das fileiras
        for(int k = 0; k<6; k++){
           buffer = buffer + matrixA[i][k] * matrixB[k][j];
        } 

        result[i][j] = buffer;

    }
    }

    for(int i = 0; i<4; i++){
        printf("\n");
    for(int j = 0; j<4; j++){
        printf("%d ", result[i][j]);
    }
    }

    return &result[0];
}

void main ()
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

    int *result = gg(matrixA, matrixB);
}