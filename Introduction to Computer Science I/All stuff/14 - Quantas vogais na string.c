//Code by Lucas Miranda - T3
#include <stdio.h>
#include <math.h>

void main(){

    for(;;){

        int a = 0, e = 0, i = 0, o = 0, u = 0;
        char entrada[20];

        printf("\nDigite seu nome: ");
        gets(entrada);
        
        for(int j = 0; entrada[j]; j++){

            printf("\n%s", entrada);

        }

        for(int j = 0; entrada[j]; j++){

            if(entrada[j] == 'a'){a++;}
            if(entrada[j] == 'e'){e++;}
            if(entrada[j] == 'i'){i++;}
            if(entrada[j] == 'o'){o++;}
            if(entrada[j] == 'u'){u++;}

        }

        printf("\n\nNo texto existem:\n%d A \n%d E \n%d I \n%d O \n%d U\n\n", a, e, i, o, u);


    }

}