#include <stdio.h>
#include <string.h>

int divisible(int data);

void main(){

    int data = 0, res = 0;

    for(;;){

        //Get
        printf("Input a number: ");
        scanf("%d", &data);
        fflush(stdin);

        //Stop if -1
        if(data == -1){break;}

        res = divisible(data);
        printf("%d\n", res);

    }
   
}

int divisible(int data){

   if(data%2 && data%3){
        if(data%5 && data%7){
            return 0;
        }else{
            return 2;
        }
   }else{
       return 1;
   }


}