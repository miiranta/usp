#include <stdio.h>

void main(){

        for(;;){

        int n = 1, linelimit = 1, current = 1;

        printf("\nInsira um valor: ");
        scanf("%d", &n);


        for(int i = 1; i<=n; i++){

            printf("%d ", i);
            current++;

            if(current >= linelimit){

                printf("\n");
                linelimit++;
                current = 0;

            }

        }

        }
    
}