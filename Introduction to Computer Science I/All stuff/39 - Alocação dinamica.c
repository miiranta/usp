#include <stdio.h>
#include <string.h>

void main(){

    //VETOR
        float *v;
        int positions = 2;

        v = (float *)calloc(positions+1, sizeof(float));


    //MATRIZ
        float **matriz;
        int i;

        //Linhas x Colunas
        int m = 3, n = 2;

        //Faz as m linhas
        matriz = (float **) calloc (m, sizeof(float *));

        //Para cada linha faz n colunas 
        for ( i = 0; i < m; i++ ) {
        matriz[i] = (float*) calloc (n, sizeof(float));	
        }


        //Alocado
        matriz[1][2] = 12; 
        printf("%f", matriz[1][2]);

        //Não alocado
        matriz[3][2] = 60; 
        printf("%f", matriz[3][2]);

    
    //FREE
    free(v);

    for (i=0; i<m; i++) {
        free(matriz[i]);
    }
    free(matriz);



}