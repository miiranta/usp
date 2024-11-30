#include <stdio.h>

main(){

    int x = 5;
    int y;

    y = x + 3;

    printf("INTS %d dividido por %d \n",y ,x);

    printf("Resultado %d",y/x);
    printf(" Resto %d \n",y%x);

    float z = 11.2;
    float w = 2;

    float h = z/w;

    printf("FLOATS z dividido por w: %f \n", h);



    int a = x++;
    printf("%d %d \n",a ,x );

    x = 5;
    int b = ++x;
    printf("%d %d \n",b ,x );



}