//Atividade Pratica 7
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#include <string.h>

#define NAME_SIZE 20
#define N         8

//Structure
typedef struct {
    int id;
    char name[NAME_SIZE];
} registration;

//Prototypes
void setArray(registration *res, int *a, char names[N+1][NAME_SIZE]);
void printArray(registration *res);
void sortBubble(registration *a, int n);

//Global Vars
int comparisons = 0;
int movements = 0;

//Start
void main(){
    
    //Setting array of struct
        registration res[N+1];

    //Populate array
        //UNCOMMENT TO CHANGE TEST CASE!
        //            s  1   2   3   4   5   6   7   8     
        int a[N+1] = {0, 45, 56, 12, 43, 95, 19,  8, 67};  
        //int a[N+1] = {0,  8, 12, 19, 43, 45, 56, 67, 95};
        //int a[N+1] = {0, 95, 67, 56, 45, 43, 19, 12,  8};
        
        char names[N+1][NAME_SIZE] = {"", "Lucas", "Ana", "Arthur", "Joca", "Renato", "Daniel", "Melissa", "Gabriel"};

        setArray(res, a, names);

    //Print array before sort
        printArray(res);

    //Sort
        sortBubble(res, N);

    //Print array after sort
        printArray(res);
        printf("\nNumber of comparisons: %d", comparisons);
        printf("\nNumber of movements: %d\n", movements);

}

void setArray(registration *res, int *a, char names[N+1][NAME_SIZE]){
    for(int i=1; i<=N; i++){    //NOTE: Starts at a[1] ends in a[N]  (a[0] is ignored)
        res[i].id = a[i];
        strcpy(res[i].name, names[i]);
    }
}

void printArray(registration *res){
    printf("\nID   NAME");
        for(int i=1; i<=N; i++){    //NOTE: Starts at a[1] ends in a[N]  (a[0] is ignored)
            printf("\n%d   ", res[i].id);
            for(int k = 0; res[i].name[k]; k++){
                printf("%c", res[i].name[k]);
            }
        }
    printf("\n");
}

void sortBubble(registration *a, int n){
    int i, j;
    registration x;

    printf("\nSorting using bubble\n");

    for(i=2; i<=n; i++){
        for(j=n; j>=i; j--){

            //Compare
            comparisons++;
            if(a[j-1].id > a[j].id){

                //Move
                movements = movements + 3;
                x = a[j-1];
                a[j-1] = a[j];
                a[j] = x;
            }

        }
    }

}

