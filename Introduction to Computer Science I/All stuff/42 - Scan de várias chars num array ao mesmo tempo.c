#include <stdio.h>
#include <string.h>

void main(){

    int array[2], i = 0;

    //Scan
    printf("Digite as ints separadas por espacos: ");
    for(i=0;i<2;i++){
        scanf("%d",&array[i]);
    }

    //Show
    for(i=0;i<2;i++){
        printf("%d ",array[i]);
    }

}


