#include <stdio.h>
#include <string.h>
#define maxSize 999

//Lucas Miranda - Número USP: 12542838

int findPosition(char string[maxSize], char subString[maxSize]);

void main(){

    //Vars
    char string[maxSize];
    char subString[maxSize];

    //prints iniciais
    printf("\n\n==========================================\n\n");
    printf("Trabalho 2 Questao 1 - Professor Zhao - Lucas Miranda Mendonca Rezende");
    printf("\n\n==========================================\n\n");

    //Get string/Substring
    printf("Insert string:\n");
    fgets(string, maxSize, stdin);
    strtok(string, "\n");

    printf("\nInsert substring to look for position:\n");
    fgets(subString, maxSize, stdin);
    strtok(subString, "\n");

    //Find position function
    int result = findPosition(string, subString);

    //Show result
    if(result == 0){
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




//Declare function
int findPosition(char string[maxSize], char subString[maxSize]){


    //Vars
    int count = 1, i = 0, j = 0;
    int stringLen = strlen(string);
    int subStringLen = strlen(subString);
    int difference = stringLen - subStringLen;

    //verify if substring is bigger then string
    if(difference < 0){ return 0; }

        //For every possible case
        for(i=0;i<=difference;i++){
            
        printf("\n==========================================\n");
        count = 0;

            printf("Testing position %d\n\n", i+1);

            //Print and test string
            for(j = 0;string[j];j++){
                printf("%c ", string[j]);

                //Shift substring +1
                if(i<=j){
                    printf("%c", subString[j-i]);

                    //Count number of matches in position
                    if(subString[j-i] == string[j]){ 
                        printf(" < MATCH");
                        count++;
                    }
                }

                printf("\n");
            }
            
            //If number of matches is substring size, return
            if(count == subStringLen){
                return i+1;
            }

        }

    //Enough matches not found
    return 0;

}

