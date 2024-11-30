#include <stdio.h>
#include <string.h>

void main(){

    //Desclarações
    char text[999], filter[100] = "AEIOUaeiou";
    int i=0, j=0, filterCount, textlen=0, filterlen=0;

    //Get string from cmd
    printf("Insert string:");
    gets(text);

   //For every filter...
   for(j = 0; filter[j]; j++){

    filterCount = 0;

        //...verify every char
        for(i = 0; text[i]; i++){

            if(text[i] == filter[j]){
            filterCount++;
            }

        }

        if(filterCount>0){
        printf("\n%c: %d", filter[j], filterCount);
        }
        
    }





}