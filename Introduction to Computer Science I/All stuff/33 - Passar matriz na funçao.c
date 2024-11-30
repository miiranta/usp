#include <stdio.h>

int gg(int matrix[6][6]){

    printf("%d ", matrix[0][3]);
    printf("%d ", matrix[2][4]);

}

void main ()
{
    int matrix[6][6] = {
    {1,2,3,4,5,9},
    {6,1,8,9,10,9},
    {11,12,1,14,15,9},
    {16,17,18,1,20,9},
    {21,22,23,24,1,9},
    {21,22,23,24,25,1}}; 

    gg(matrix);

}