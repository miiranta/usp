#include <stdio.h>
#include <string.h>

int dataRead(int data);

void main(){

    int data = 0;

    for(;;){
        printf("\n");

        //Get
        printf("\nSet a number: \n");
        scanf("%d", &data);
        fflush(stdin);

        //Is -1 or negative?
        if(data == -1){break;}
        if(data < -1){continue;}
    
        dataRead(data);

    }
}


int dataRead(int data){

int res = 1, i = 0;

        //Factorial
        if(data <= 7){
            res = 1;
            if(data != 0){
                for(i = 1; i<=data; i++){
                    res = res*i;
                }
            }
            else if(data == 0){res = 1;}
            printf("The factorial is %d", res);

        //Sum
        }else{
            res = 0;
            for(i = 1; i<=data; i++){
                res = res+i;
            }
            printf("The sum from 1 to %d is %d",data, res);
        }

}