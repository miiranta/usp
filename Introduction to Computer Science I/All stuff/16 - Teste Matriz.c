#include <stdio.h>


void main(){

    char a[10][10] = {"a","b","Lucas","Maria", "Joao", "Angelololo"};

    for(int i = 0;i<10;i++){
        printf("\n");
    for(int j = 0;j<10;j++){
        
        printf("%c", a[i][j]);


    }
    }

    printf("%c", a[2][2]);

}