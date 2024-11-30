#include <stdio.h>


main(){

    int var1 = 5, var2= 10;
    int aux;

    //O que não fazer
    //var1 = var2;
    //var2 = var1;

    //O que fazer
    aux = var1;
    var1 = var2;
    var2 = aux;

    printf("%d %d", var1, var2);

}