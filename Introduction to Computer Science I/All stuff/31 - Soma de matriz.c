#include <stdio.h>

int functionThatSumsTheElementsOfaFiveByFiveMatrix(int *matrix[0]){

    int add = 0;
    int *p = &matrix[0];

    for(int i = 0; i<25; i++){
       add = add + *p;
       p++;
    }

    return add;
   
}

void main ()
{
    int matrix[5][5] = {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}}; 
    int add = functionThatSumsTheElementsOfaFiveByFiveMatrix(matrix);

    printf("%d", add);

}