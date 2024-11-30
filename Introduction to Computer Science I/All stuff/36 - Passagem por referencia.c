#include <stdio.h>

void gg(int matrix[2][2], int matrixR[2][2]){

    matrixR[1][1] = matrix[0][0];

}

main()
{
    int matrix[2][2] = {
    {98,2},
    {3,4}
    }; 

    int matrixR[2][2]= {
    {0,0},
    {0,0}
    };

    gg(matrix, matrixR);

    printf("%d", matrixR[1][1]);



}