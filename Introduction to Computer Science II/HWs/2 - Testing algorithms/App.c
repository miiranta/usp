//Atividade Pratica 2
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double* prefixMedia1(double* X, int n);
double* prefixMedia2(double* X, int n);

void main(){
    clock_t time1, time2, time3, time4;
    double totaltime, totaltime2;

    //Vector Size
    int n = 100000; 

    //Creates vector 
    double *X;
    X = (double*)calloc(n+1, sizeof(double));
    for(int i = 0; i<n; i++){
        X[i] = i;
    }

    //Timer 1
    printf("N = %d\n", n);

    time1 = clock();
    double *A = prefixMedia1(X, n);
    time2 = clock();

    totaltime = difftime(time2, time1) / CLOCKS_PER_SEC;
    printf("Time 1: %lf \n", totaltime);

    //Timer 2
    time3 = clock();
    double *B = prefixMedia2(X, n);
    time4 = clock();

    totaltime2 = difftime(time4, time3) / CLOCKS_PER_SEC;
    printf("Time 2: %lf \n", totaltime2);

}

double* prefixMedia1(double* X, int n){
    int i = 0, j = 0;
    double a = 0;
    double *A = (double*)calloc(n+1, sizeof(double));

    for(i = 0; i<n; i++){
        a = 0;
        for(j = 0; j<=i; j++){
            a = a + X[j];
        }
        A[i] = a/((double)(i+1));
    }

    return A;
}

double* prefixMedia2(double* X, int n){
    int i = 0;
    double s = 0;
    double *A = (double*)calloc(n+1, sizeof(double));

    for(i = 0; i<n; i++){
        s = s + X[i];
        A[i] = s/((double)(i+1));
    }

    return A;
}
