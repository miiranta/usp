#include <stdio.h>
#include <string.h>

int mult(int n, int k);

void main(){

int k = 0, n = 0;

//Get int
printf("Input positive int:\n ");
scanf("%d", &k);
fflush(stdin);

//Get number of sums
printf("How many multiples to sum?\n ");
scanf("%d", &n);
fflush(stdin);

//Verify positive
if(n <= 0 || k <= 0){return;}

mult(n, k);

}

int mult(int n, int k){

    int i = 0, multiple = 1, sum = 0;

    //Print multiples
    printf("\nMultiples: \n");
    for(i = 0; i<n; i++){
        multiple = k*i;
        printf(" %d", multiple);
        sum = sum + multiple;
    }

    //Print sum
    printf("\nSum of multiples:\n %d", sum);

}
