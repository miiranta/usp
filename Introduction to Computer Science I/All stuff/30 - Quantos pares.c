#include <stdio.h>

int *pairs(int *matrix){

    static int response[15];
    int *p = response;

    for(int i = 0; i<15; i++){
       if(matrix[i]%2){
           continue;
       }else{
           *p = matrix[i];
           p++;
           printf("%d ", matrix[i]);
       }
    }

    return response;
   
}

void main ()
{

    //vars
    int matrix[15] = {1,2,3,4,5,6,7,8,9,10,11,12,17,56,28}; 
    
    int *response;
    response = pairs(matrix);
    
}