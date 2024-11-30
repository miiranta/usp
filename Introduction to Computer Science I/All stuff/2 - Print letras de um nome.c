#include <stdio.h>

void main(){

int index = 0, contador;
char letras[4] = "Joao";

    for (contador = 0; contador < 10; contador++)
    {
        printf("%c\n", letras[index]);
        index = (index == 3) ? index = 0: ++index;
    }

}