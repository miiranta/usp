#include <stdio.h>


main(){

    int count;
    char ch;

    for(;;){

        ch = getch();

        if(ch != NULL){
            printf("%c \n", ch);

            break;
        }
        

    }

}