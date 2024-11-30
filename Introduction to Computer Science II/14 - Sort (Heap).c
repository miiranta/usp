#include <stdio.h>
void sort(int* a, int N);
void heapify(int* a, int L, int R);

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
    int L, R, w;
    
    for(L=N/2; L>=1; L=L-1){
        heapify(a, L, N);
    }

    for(R=N; R>=2; R=R-1){
        w = a[1];
        a[1] = a[R];
        a[R] = w;
        heapify(a, 1, R-1);
    }
}

void heapify(int* a, int L, int R){
    int i=L, j=2*L, x=a[L];

    if((j<R)&&(a[j]<a[j+1])){
        j = j + 1;
    }

    while((j<=R)&&(x<a[j])){
        a[i] = a[j];
        i = j;
        j = 2*j;
        if((j<R)&&(a[j]<a[j+1])){
            j = j + 1;
        }
    }

    a[i] = x;
}

