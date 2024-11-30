#include <stdio.h>
#include <string.h>

#define MAX_STRING_SIZE 11
#define MAX_MONTH_SIZE 20000

//Struct
typedef struct{
    char code[MAX_STRING_SIZE];
    int month;
} Month;

//Sorting and Counting Functions
#include "Sorts.h"

//Prototype
int readFile(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month, int monthNumber, int arrayStartPoint);
void writeFile(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month, int totalArraySize);
void printArray(int totalArraySize, Month codes[MAX_MONTH_SIZE]);
void cloneArray(int totalArraySize, Month codes[MAX_MONTH_SIZE], Month codesTemp[MAX_MONTH_SIZE]);
void testSorts(int totalArraySize, Month codes[MAX_MONTH_SIZE]);

//Main
void main(){

    //Set
    char inputDirectory[100] = "input/mes_1.txt"; //For ONE FILE
    char outputDirectory[100] = "output/mes_1.txt";
    int monthSet = 1;
    
    //Vars
    FILE *fpointer;
    Month codes[MAX_MONTH_SIZE];
    int totalArraySize = 0;

    //Read files and make array
        //FOR MULTIPLE FILES! UNCOMMENT to select
        //totalArraySize = readFile(fpointer, codes, "input/mes_1.txt", 1, totalArraySize) + totalArraySize;
        //totalArraySize = readFile(fpointer, codes, "input/mes_2.txt", 2, totalArraySize) + totalArraySize;
        //totalArraySize = readFile(fpointer, codes, "input/mes_3.txt", 3, totalArraySize) + totalArraySize;
        //totalArraySize = readFile(fpointer, codes, "input/mes_4.txt", 4, totalArraySize) + totalArraySize;
        //totalArraySize = readFile(fpointer, codes, "input/mes_5.txt", 5, totalArraySize) + totalArraySize;

        //FOR ONE FILE! UNCOMMENT to select
        totalArraySize = readFile(fpointer, codes, inputDirectory, monthSet, totalArraySize) + totalArraySize;

    //Test and show comp/moves for each sort
        testSorts(totalArraySize, codes);
        
    //Sort main array
        sortQuick(codes, totalArraySize);
    
    //Prints array (DEBUG)
        //printArray(totalArraySize, codes);

    //Write array
        writeFile(fpointer, codes, outputDirectory, totalArraySize);
        
}

int readFile(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month, int monthNumber, int arrayStartPoint){
    int loop = 1, i;
    char dummy;

    //Read
    fpointer = fopen(month, "rt+");
    for(i=1; loop; i++){
        fgets(codes[i + arrayStartPoint].code, MAX_STRING_SIZE, fpointer);
        codes[i + arrayStartPoint].month = monthNumber;
        fscanf(fpointer, "\n", &dummy);

        if(feof(fpointer)){loop = 0;};
    }

    fclose(fpointer);

    //Returns Array Size
    return i-1;
}

void writeFile(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month, int totalArraySize){
    int j;

    //Write
    fpointer = fopen(month, "wt+");
    for(j=1; totalArraySize>=j; j++){
        fputs(codes[j].code, fpointer);
        fprintf(fpointer, " %d\n", codes[j].month); 
    }

    fclose(fpointer);
}

void printArray(int totalArraySize, Month codes[MAX_MONTH_SIZE]){
    int j, k;

    for(j=1; totalArraySize>=j; j++){
        printf("%d - ", j);
        for(k = 0; codes[j].code[k]; k++){
            printf("%c", codes[j].code[k]);
        }
        printf(" - %d\n", codes[j].month);
    }
}

void cloneArray(int totalArraySize, Month codes[MAX_MONTH_SIZE], Month codesTemp[MAX_MONTH_SIZE]){
    int j;
    
    for(j=1; totalArraySize>=j; j++){
        strcpy(codesTemp[j].code, codes[j].code);
        codesTemp[j].month = codes[j].month;
    }
}

void testSorts(int totalArraySize, Month codes[MAX_MONTH_SIZE]){
    Month codesTemp[MAX_MONTH_SIZE];
    
    cloneArray(totalArraySize, codes, codesTemp);
    sortDirectInsertion(codesTemp, totalArraySize);
    showCount("Direct Insertion");
        
    cloneArray(totalArraySize, codes, codesTemp);
    sortBinaryInsertion(codesTemp, totalArraySize); 
    showCount("Binary Insertion");
     
    cloneArray(totalArraySize, codes, codesTemp);
    sortSelection(codesTemp, totalArraySize); 
    showCount("Selection");
    
    cloneArray(totalArraySize, codes, codesTemp);
    sortBubble(codesTemp, totalArraySize);  
    showCount("Bubble");
    
    cloneArray(totalArraySize, codes, codesTemp);
    sortShaker(codesTemp, totalArraySize); 
    showCount("Shaker");
      
    cloneArray(totalArraySize, codes, codesTemp);
    sortFusion(codesTemp, totalArraySize); 
    showCount("Fusion");
    
    cloneArray(totalArraySize, codes, codesTemp);
    sortHeap(codesTemp, totalArraySize);  
    showCount("Heap");
    
    cloneArray(totalArraySize, codes, codesTemp);
    sortQuick(codesTemp, totalArraySize);   
    showCount("Quick");
    
}