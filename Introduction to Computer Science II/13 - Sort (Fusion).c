#include <stdio.h>
void sort(int* a, int N);
void mpass(int* a, int N, int p, int* c);
void merge(int* a, int L, int h, int R, int* c);

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
    int p = 1;
    int c[N+1];

    while(p<N){
        mpass(a, N, p, c);
        p = 2*p;
        mpass(c, N, p, a);
        p = 2*p;
    }
}

void mpass(int* a, int N, int p, int* c){
    int i = 1, j;

    while(i<=N-2*p+1){
        merge(a, i, i+p-1, i+2*p-1, c);
        i = i + 2*p;
    }

    if(i+p-1<N){
        merge(a, i, i+p-1, N, c);
    }else{
        for(j=i; j<=N; j++){
            c[j] = a[j];
        }
    }
}

void merge(int* a, int L, int h, int R, int* c){
    int i=L, j=h+1, k=L-1;

    while((i<=h)&&(j<=R)){
        k = k + 1;
        if(a[i]<a[j]){
            c[k] = a[i];
            i = i + 1;
        }else{
            c[k] = a[j];
            j = j + 1;
        }
    }

    while(i<=h){
        k = k + 1;
        c[k] = a[i];
        i = i + 1;
    }

    while(j<=R){
        k = k + 1;
        c[k] = a[j];
        j = j + 1;
    }

}




