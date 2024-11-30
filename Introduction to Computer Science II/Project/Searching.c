#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_STRING_SIZE 11
#define MAX_MONTH_SIZE 20000

//Struct
typedef struct{
    char code[MAX_STRING_SIZE];
    int month;
} Month;

//Prototype
int readOutput(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month);
int search(Month* a, int N, char* x);
void printArray(int totalArraySize, Month codes[MAX_MONTH_SIZE]);

//Main
void main(){

    FILE *fpointer;
    Month codes[MAX_MONTH_SIZE], codes1[MAX_MONTH_SIZE], codes2[MAX_MONTH_SIZE], codes3[MAX_MONTH_SIZE], codes4[MAX_MONTH_SIZE], codes5[MAX_MONTH_SIZE];
    int arraySize1 = 0, arraySize2 = 0, arraySize3 = 0, arraySize4 = 0, arraySize5 = 0, arraySizeAll = 0, loop = 1, res = -1, month = 0;
    char x[MAX_STRING_SIZE];

    //Read 
        arraySize1 = readOutput(fpointer, codes1, "output/mes_1.txt");
        arraySize2 = readOutput(fpointer, codes2, "output/mes_2.txt");
        arraySize3 = readOutput(fpointer, codes3, "output/mes_3.txt");
        arraySize4 = readOutput(fpointer, codes4, "output/mes_4.txt");
        arraySize5 = readOutput(fpointer, codes5, "output/mes_5.txt");
 
    //Prints array (DEBUG)
        //printArray(totalArraySize, codes1);

    //Search
        //Loop
        do{
            res = -1;
            month = 0;

            //Get search key
            printf("\nInsert search key (the code you wanna find): \n");
            fgets(x, MAX_STRING_SIZE, stdin);
            
            //Search next month if not found
            res = search(codes1, arraySize1, x);
            month = 1;
            if(res == -1){
                res = search(codes2, arraySize2, x);
                month = 2;
                if(res == -1){
                    res = search(codes3, arraySize3, x);
                    month = 3;
                    if(res == -1){
                        res = search(codes4, arraySize4, x);
                        month = 4;
                        if(res == -1){
                            res = search(codes5, arraySize5, x);
                            month = 5;
                        }
                    }   
                }
            }
                
            //Print code info
            if(res != -1){
                printf("============\n");
                printf("Code found at index %d\n", res);
                switch(month){
                    case 1:
                        printf(" Code: %s\n", codes1[res].code);
                        break;
                    case 2:
                        printf(" Code: %s\n", codes2[res].code);
                        break;
                    case 3:
                        printf(" Code: %s\n", codes3[res].code);
                        break;
                    case 4:
                        printf(" Code: %s\n", codes4[res].code);
                        break;
                    case 5:
                        printf(" Code: %s\n", codes5[res].code);
                        break;
                }
                printf(" Month: %d\n", month);
                printf("============\n");
            }else{
                printf("============\n");
                printf("Code not found!\n");
                printf("============\n");
            }

            fflush(stdin);
        }while(loop);

}

int search(Month* a, int N, char* x){
    int L = 0;
    int R = N-1;
    int Success = 0;
    int m;

    while((L<=R)&&(Success == 0)){
        m = floor((R+L)/2);

        if(!(strcmp(a[m].code, x))){ Success = 1;}
        else{
            if(strcmp(a[m].code, x)<0){
                L = m + 1;
            }else{
                R = m - 1;
            }
        }
    }

    if(Success == 0){
        return -1;
    }

    return m;
}

int readOutput(FILE *fpointer, Month codes[MAX_MONTH_SIZE], char* month){
    int loop = 1, i;
    char dummy;

    //Read
    fpointer = fopen(month, "rt+");
    for(i=1; loop; i++){
        fgets(codes[i].code, MAX_STRING_SIZE, fpointer);
        fscanf(fpointer, " %d", &codes[i].month);
        fscanf(fpointer, "\n", &dummy);

        if(feof(fpointer)){loop = 0;};
    }

    fclose(fpointer);

    //Returns Array Size
    return i;
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

