#include <stdio.h>
#define max 20

int anagrama(char *string1, char *string2){

    int i, j;
    int len1 = strlen(string1);
    int len2 = strlen(string2);

    if(len1 != len2){
        return 0;
    }

    for(i = 0;i<len1;i++){
        for(j = 0;j<len2;j++){
            if(string1[i] == string2[j]){
                string2[j] = 0;
                break;
            }
        }
    }

    for(j = 0;j<len2;j++){
        if(string2[j] != 0){
            return 0;
        };
    }

    return 1;

}


void main(){

char string1[max] = "amor";
char string2[max] = "roma";

int response = anagrama(string1, string2);

if(response == 1){
    printf("Its an anagram!");
}else{
    printf("Its NOT an anagram!");
}

}