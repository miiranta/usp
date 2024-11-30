//Atividade Pratica 3
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#define N 10

int findX(int* v, int n, int x);

void main(){

    //Search word
    int x = 23;

    //Vector
    int v[N] = {1,2,3,4,5,6,7,8,23};

    //Print
    printf("%d", findX(v, N, x));

}

int findX(int* v, int n, int x){
    int i = 0;

    for(i = 0; i<n; i++){
        if(v[i] == x){
            return v[i];
        }
    }
    return -1;

}