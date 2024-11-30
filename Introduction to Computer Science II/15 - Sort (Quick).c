#include <stdio.h>
#include <math.h>
void sort(int* a, int N);
void qsort(int* a, int L, int R);

void main(){
    //           s  1  2   3   4   5   6   7   8  9    10     (NO SENTINEL! s is ignored)
    int a[11] = {0, 9, 50, 40, 20, 30, 30, 10, 5, 392, 23};   //Has to be declared N+1
    int N = 10;

    sort(a, N);

    for(int i=1; i<=N; i++){  //NOTE: Starts at a[1] ends in a[N]  (a[0] is ignored)
        printf("%d ", a[i]);
    }

}


void sort(int* a, int N){
    qsort(a, 1, N);
}

void qsort(int* a, int L, int R){
    int fl = floor((L+R)/2);
    int w, i=L, j=R, x=a[fl];

    do{

        while(a[i]<x){
            i = i + 1;
        }

        while(x<a[j]){
            j = j - 1;
        }

        if(i<=j){
            w = a[i];
            a[i] = a[j];
            a[j] = w;
            i = i + 1;
            j = j -1;
        }


    }while(i<=j);

    if(L<j){
        qsort(a, L, j);
    }

    if(i<R){
        qsort(a, i, R);
    }

}