#include <stdio.h>

int functionToFindTheSumOfTheDiagonalOfaSixBySixMatrix(int *matrix[0]){

    int addPrincipal = 0, addSecundaria = 0;
    int *p = &matrix[0];

    //Principal
    for(int i = 0; i<36; i++){
        if(i%7){
            continue;
        }else{
            addPrincipal = addPrincipal + *(p+i);
            printf("%d ", *(p+i));
        }
    }
    printf("\nSoma da principal: %d\n\n", addPrincipal);

    //Secundaria
    for(int i = 0; i<30; i++){
        if((i+5)%5){
            continue;
        }else{
            addSecundaria = addSecundaria + *(p+i+5);
            printf("%d ", *(p+i+5));
        }
    }
    printf("\nSoma da secundaria: %d", addSecundaria);
    
    return 0;
   
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

    int add = functionToFindTheSumOfTheDiagonalOfaSixBySixMatrix(matrix);
}