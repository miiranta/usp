#include <stdio.h>

int gg(int matrix[7][6]){

    int addLine = 0, addColumn = 0;

    for(int i = 0; i<7; i++){
    for(int j = 0; j<6; j++){
        if(i == 4){
            addLine = addLine + matrix[i][j];
        }
        if(j == 2){
            addColumn = addColumn + matrix[i][j];
        }
    }
    }

    printf("\n\nSum of line 5: %d", addLine);
    printf("\nSum of column 3: %d", addColumn);

    return 0;

}

void main ()
{
    int matrix[7][6] = {
    {1,2,3,4,5,9},
    {6,1,8,9,10,9},
    {11,12,1,14,15,9},
    {16,17,18,1,20,9},
    {21,22,23,24,1,9},
    {21,22,23,24,25,1},
    {21,22,23,24,25,1}
    }; 

    gg(matrix);
}