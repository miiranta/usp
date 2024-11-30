#include <stdio.h>
#include <string.h>

void main(){

    char name[100];
    int i, len;

    printf("Insert name: ");
    gets(name);

    len = strlen(name);
    for(i = 0; name[i]; i++){
        printf("%c", name[len-i-1]);
    }


}