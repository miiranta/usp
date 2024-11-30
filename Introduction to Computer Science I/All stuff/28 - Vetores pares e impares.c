#include <stdio.h>

int pairs(int *p) {

    int odd[15];
    int even[15];
    int *oddp = &odd[0];
    int *evenp = &even[0];

    for(int i = 0; i<30; i++){
        if(*p%2 || *p==0){
            //Not divisible
            *oddp = *p;
            p++;
            oddp++;
        }else{
            //Divisible by 2
            *evenp = *p;
            p++;
            evenp++;
        }
    }

    printf("\nOdd numbers: ");
    for(int j = 0; j<15; j++){
        printf("%d ", odd[j]);
    }

    printf("\nEven numbers: ");
    for(int k = 0; k<15; k++){
        printf("%d ", even[k]);
    }

}


int main(){

int vector[30] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
int *p = &vector[0];

pairs(p);

}