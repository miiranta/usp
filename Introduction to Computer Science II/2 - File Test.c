#include <stdio.h>

void main(){
    int count = 0;

    //Creates File pointer
    FILE *fpointer;


    //Creates and/or opens file "File.txt" from the same directory
    //fpointer is now a pointer for that file
    fpointer = fopen("2 - File.txt","rt+");


    //Reads file to find INT --> Count
    fscanf(fpointer,"%d", &count);
    //OR
    //fgets(charVar, MAXSIZE, fpointer);


    //Cleans File
    fpointer = fopen("2 - File.txt","wt+");


    //Rewrites INT with plus one
    fprintf(fpointer, "%d", count + 1);
    printf("%d", count + 1);


    //fpointer will no longer be a pointer for that file
    fclose(fpointer);


}