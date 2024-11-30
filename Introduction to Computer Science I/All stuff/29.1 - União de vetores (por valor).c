#include <stdio.h>

int* unionM(int matrixOne[10], int matrixTwo[10]){

    static int unionM[20];
    int *p = &unionM[0];
    int already = 0;

    //MatrixOne
    for(int i = 0; i<10; i++){

        already = 0;
        for(int j = 0; j<20; j++){
            if(unionM[j] == matrixOne[i]){
                already = 1;
                continue;
            }
        }
        if(already == 0){
            *p = matrixOne[i];
            p++;
        }
        
    }

    //MatrixTwo
    for(int i = 0; i<10; i++){

        already = 0;
        for(int j = 0; j<20; j++){
            if(unionM[j] == matrixTwo[i]){
                already = 1;
                continue;
            }
        }
        if(already == 0){
            *p = matrixTwo[i];
            p++;
        }
        
    }

    return &unionM;
   
}


void main ()
{

    //vars
    int matrixOne[10] = {1,2,3,4,5,6,7,8,9,10}; 
    int matrixTwo[10] = {1,2,3,4,5,6,7,8,11,12}; 
    
    int *response = unionM(matrixOne, matrixTwo);
    
    printf("Union = { ");
    for(int i = 0; i<20; i++){
        printf("%d ", response[i]);
    }
    printf("}");

}