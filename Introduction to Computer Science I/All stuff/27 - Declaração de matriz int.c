#include <stdio.h>

void main ()
{

    //vars
    int *p; //Aqui
    int i, j;
    int size = 4;
    int matrix[4][4] ={{1,2,3,4},{5,6,7,8},{1,2,3,4},{5,6,7,8}}; 
    int count = 0;

   
    //Print elements
    for(i=0; i<size; i++){
        printf("\n");
        for(j=0; j<size; j++){
        printf("%d  ", matrix[i][j]);
        }
    }


}