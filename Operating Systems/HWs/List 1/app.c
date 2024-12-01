#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void main(int argc, char* argv[]){

    int my_array[10];

    if(argc != 11){
        printf("Please write 10 ints as parameters!\n");
        return;
    }

    for(int i = 0; i<argc-1; i++){
        my_array[i] = atoi(argv[i+1]);
    }

    for(int j = 0; j<argc-1; j++){
        printf("%d - %d\n", j, my_array[j]);
    }


    //Start counting
    clock_t c2, c1;
    float tmp;
    c1 = clock();

    //Fork
    pid_t pid;
    pid = fork();

    if (pid < 0) { /* erro */
        fprintf(stderr, "Falha no Fork ");
        exit(-1);
    } else if (pid == 0) { /* processo filho */

        quickSort(my_array, 0, 9);

        //End counting
        c2 = clock();
        tmp = (c2 - c1)*1000/CLOCKS_PER_SEC; 
        printf("FILHO: %f", tmp);

        exit(1);
    } else { /* processo pai */

        simpleSort(my_array, 10);

        //End counting
        c2 = clock();
        tmp = (c2 - c1)*1000/CLOCKS_PER_SEC; 
        printf("PAI: %f", tmp);

        wait(NULL); /* pai espera o término do filho */
        exit(0);
    }


}

void quickSort(int valor[], int esquerda, int direita){
    int i, j, x, y;
    i = esquerda;
    j = direita;
    x = valor[(esquerda + direita) / 2];

    while(i <= j){
        while(valor[i] < x && i < direita){
            i++;
        }
        while(valor[j] > x && j > esquerda){
            j--;
        }
        if(i <= j){
            y = valor[i];
            valor[i] = valor[j];
            valor[j] = y;
            i++;
            j--;
        }
    }

    if(j > esquerda){
        quickSort(valor, esquerda, j);
    }
    if(i < direita){
        quickSort(valor, i, direita);
    }
}

void simpleSort(int valor[], int tamanho){
    int i, j;
    int aux;

    for(i = 0; i < tamanho; i++){
        for(j = i+1; j < tamanho; j++){
            if(valor[i] > valor[j]){
                aux = valor[i];
                valor[i] = valor[j];
                valor[j] = aux;
            }
        }
    }
}