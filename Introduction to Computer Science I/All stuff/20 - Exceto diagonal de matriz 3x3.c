#include <stdio.h>
#include <string.h>

void main(){

//vars
int i, printdiagonal=3, breakrow=0;
char matrix[10] = "abcdefghi"; 

//Print diagonal
for(i = 0; matrix[i]; i++){

    printdiagonal++;
    breakrow++;

    //Print if position is not diagonal
    if(printdiagonal>3){
        printdiagonal = 0;
        printf("   ");
    }
    else{
        printf("%c  ", matrix[i]);
    }

    //Break line
    if(breakrow>2){
        breakrow = 0;
        printf("\n");
    }

    
   
}


}