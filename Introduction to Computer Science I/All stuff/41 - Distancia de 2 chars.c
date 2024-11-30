#include <stdio.h>
#include <string.h>

int readChar(char a, char b);

void main(){

    char a = "a", b = "a", res = 0;

    //A
    printf("Input the first character: ");
    scanf("%c", &a);
    fflush(stdin);

    //B
    printf("Input the second character: ");
    scanf("%c", &b);
    fflush(stdin);

    res = readChar(a, b);

    printf("\n");
    if(res != -1){
        printf("Distance between chars is %d", res);
    }else{
        printf("Chars need to be in alphabetic order!");
    }

}

int readChar(char a, char b){

    a = tolower(a);
    b = tolower(b);

    int distance = b - a;

    if(distance >= 0){
        return distance;
    }else{
        return -1;
    }


}