#include <stdio.h>
#include <string.h>
#define maxSize 999

int findPosition(char string[maxSize], char subString[maxSize]);

void main(){

    char string[maxSize];
    char subString[maxSize];

    printf("Insert string:\n");
    fgets(string, maxSize, stdin);
    strtok(string, "\n");

    printf("\nInsert substring to look for position:\n");
    fgets(subString, maxSize, stdin);
    strtok(subString, "\n");

    int result = findPosition(string, subString);

    if(result == -1){
        printf("\n==========================================\n");
        printf("Substring NOT FOUND!");
        printf("\n==========================================\n\n");
    }
    else{
        printf("\n==========================================\n");
        printf("Substring FOUND in position number %d", result);
        printf("\n==========================================\n\n");
    }

}

int findPosition(char string[maxSize], char subString[maxSize]){

    int count = 1, i = 0, j = 0;
    int stringLen = strlen(string);
    int subStringLen = strlen(subString);
    int difference = stringLen - subStringLen;

    if(difference < 0){ return 0; }

        for(i=0;i<=difference;i++){
            
        printf("\n==========================================\n");
        count = 0;

            printf("Testing position %d\n\n", i);

            for(j = 0;string[j];j++){
                printf("%c ", string[j]);

                if(i<=j){
                    printf("%c", subString[j-i]); 

                    if(subString[j-i] == string[j]){ 
                        printf(" < MATCH");
                        count++;
                    }
                }

                printf("\n");
            }
            
            if(count == subStringLen){
                return i;
            }

        }

    return -1;

}

