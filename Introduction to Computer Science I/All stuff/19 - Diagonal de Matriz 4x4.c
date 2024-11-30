#include <stdio.h>
#include <string.h>

void main(){

//vars
int i, breakrow=0, printdiagonal=4;
char matrix[17] = "abcdefghijklmnop"; 


//Print matrix
for(i = 0; matrix[i]; i++){

    printf("%c  ", matrix[i]);
    breakrow++;

    //Break line
    if(breakrow>3){
        breakrow = 0;
        printf("\n");
    }
   
}

//Separate
printf("\n");

//Print diagonal
for(i = 0; matrix[i]; i++){

    printdiagonal++;

    if(printdiagonal>4){
        printdiagonal = 0;
        printf("%c  ", matrix[i]);
    }
   
}


}