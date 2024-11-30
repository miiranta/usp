#include <stdio.h>


main(){

    int a = 0, aux = 0, count = 0;

    for(count = 1; count<=100; count++){

        

        if(aux == 0){
        aux = 1;
        a = a + count;

        printf("Soma %d = %d \n", count, a);

        continue;

        }

        if(aux == 1){
        aux = 0;
        a = a - count;

        printf("Subtrai %d = %d \n", count, a);

        continue;

        }

        

    }

}