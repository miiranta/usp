//Atividade Pratica 4
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#define MAXSIZE 100

void transpose_Matrix(int n, float mA[MAXSIZE][MAXSIZE]);
void multiply_Matrices(int n, float mA[MAXSIZE][MAXSIZE], float mB[MAXSIZE][MAXSIZE]);
void trace_Matrix(int n, float mA[MAXSIZE][MAXSIZE]);

void main(){

//A
int na = 3; //Number of lines/columns A
float mA[MAXSIZE][MAXSIZE] = {
    {1,2,3},
    {4,5,6},
    {7,8,9}
};

//B
int nb = 3; //Number of lines/columns B
float mB[MAXSIZE][MAXSIZE] = {
    {2,2,3},
    {4,25,6},
    {7,8,9}
};

//Function call
transpose_Matrix(na, mA);
printf("\n");

multiply_Matrices(na, mA, mB);
printf("\n\n");

trace_Matrix(nb, mB);

}

void transpose_Matrix(int n, float mA[MAXSIZE][MAXSIZE]){
    int i, j;
    for(i=0; i<n; i++){
        printf("\n");
        for(j=0; j<n; j++){
            printf("%f ", mA[j][i]);
        }
    }
}

void multiply_Matrices(int n, float mA[MAXSIZE][MAXSIZE], float mB[MAXSIZE][MAXSIZE]){
    int i, j, k;
    float buffer;
    for(i=0; i<n; i++){
        printf("\n");
        for(j=0; j<n; j++){
            buffer = 0;
            for(k=0; k<n; k++){
                buffer = mA[i][k]*mB[k][j] + buffer;
            }
            printf("%f ", buffer);
        }
    }

}

void trace_Matrix(int n, float mA[MAXSIZE][MAXSIZE]){
    int i; 
    float buffer = 0;
    for(i=0; i<n; i++){
        buffer = buffer + mA[i][i];
    }

    printf("%f", buffer);
}

