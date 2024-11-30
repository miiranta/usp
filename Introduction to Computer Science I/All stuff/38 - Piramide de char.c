#include <stdio.h>
#include <string.h>

void main(){

char string1[20] = "gg";

int len = strlen(string1);
int i,j,k, count = 0;

for(k=0;k<len;k++){
    printf("\n");
    count++;
    for(j=0;j<count;j++){
        for(i = 0;i<len;i++){
            printf("%c",string1[i]);
        }
        printf(" ");
    }
}

}