#include <stdio.h>
void main ()
{
int count;
char str1[100] = "AAA", str2[100];


for (count = 0; str1[count]; count++)
{

    str2[count] = str1[count];
    printf("%d ", count);

}

printf("\n%d ", count);

str2[count] = '\0';

}

