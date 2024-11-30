#include <stdio.h>
#include <string.h>

int div(int n, int k);

void main(){

int k = 0, n = 0;

printf("Input positive int:\n ");
scanf("%d", &k);
fflush(stdin);

printf("How many divisors to find?\n ");
scanf("%d", &n);
fflush(stdin);

if(n <= 0 || k <= 0){return;}

div(n, k);

}

int div(int n, int k){

    int i = 0, count = 0;
    int divisors[k];

    for(i = 1; count < n && i <= k; i++){

        if(k%i){
            continue;
        }else{
            divisors[count] = i;
            count++;
        }

    }

    printf("Divisors: \n");
    for(i = 0; i<count; i++){
        printf(" %d", divisors[i]);
    }

}

