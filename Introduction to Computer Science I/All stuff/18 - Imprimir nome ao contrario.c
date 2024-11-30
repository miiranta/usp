#include <stdio.h>
#include <string.h>

void main(){

    //Vars
    char name[100];
    int i, len;

    //Get name
    printf("Insert name: ");
    gets(name);

    //Name lenght
    len = strlen(name);

    for(i = 0; name[i]; i++){

        printf("%c", name[len-i-1]);

    }





}