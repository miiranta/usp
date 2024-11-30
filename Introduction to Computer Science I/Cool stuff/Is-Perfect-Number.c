#include <stdio.h>
#include <math.h>

int perfect(int num) {

    int add = 0;

    for(int i=1;i<num;i++){
        if(num%i>=1){
            continue;
        }else{
            add = i + add;
            printf("%d ", add);
        }
    }
    
    if(add == num){
        return 1;
    }else{
        return 0;
    }

}


int main(){

//vars
int number = 0;

    for(;;){

        //Get elements
        printf("\n\nIs it perfect? Insert:\n");
        scanf("%d", &number);
        fflush(stdin);

        if(perfect(number)){
            printf("\nYES\n");
        }else{        
            printf("\nNO\n");
        }

    }

    

}