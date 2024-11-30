//Atividade Pratica 6
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#include <string.h>

#define SIZE 10 //Amount of stuff in the file!!
#define ADDRESS_SIZE 100

//Structure
typedef struct {
    char address[ADDRESS_SIZE];
    int monthly_earnings;
    int number_of_people;
    int number_of_people_school_age;
} research;

//Prototypes
void sort(research* a, int N);
void writeSentinel(research* data);

void main(){

    research data[SIZE+1];
    FILE *fpointer;

    //Read
    fpointer = fopen("data.txt","rt+");
    printf("\nRead: \n");
    for(int i=1; i<SIZE+1; i++){
        char dummy;
        fgets(data[i].address, ADDRESS_SIZE, fpointer);
        fscanf(fpointer, "%d", &data[i].monthly_earnings);
        fscanf(fpointer, "%d", &data[i].number_of_people);
        fscanf(fpointer, "%d", &data[i].number_of_people_school_age);
        fscanf(fpointer, "\n\n", &dummy);
        
        printf(" %c, %d, %d, %d\n", data[i].address[0], data[i].monthly_earnings, data[i].number_of_people,  data[i].number_of_people_school_age);
    }

    //Sort
    writeSentinel(data);
    sort(data, SIZE);

    //Write
    fpointer = fopen("data.txt","wt+");
    
    printf("\nWrite: \n");
    for(int i=1; i<SIZE+1; i++){
        if(!strcmp(data[i].address, "")){
            strcpy(data[i].address, "none");
        }
        data[i].address[strcspn(data[i].address, "\n")] = 0;
        fputs(data[i].address, fpointer);
        fprintf(fpointer, "\n%d\n%d\n%d\n", data[i].monthly_earnings, data[i].number_of_people,  data[i].number_of_people_school_age);   

        printf(" %c, %d, %d, %d\n", data[i].address[0], data[i].monthly_earnings, data[i].number_of_people,  data[i].number_of_people_school_age);
    }
    

    //File CLOSE
    fclose(fpointer);

}

//"Direct" Insertion
void sort(research* data, int N){
    int i, j;
    research x;

    for(i = 2; i<=N; i++){

        x = data[i];
        data[0] = x;
        j = i;

        //Linear search with sentinel
        while(strcmp(data[j-1].address, x.address) > 0){
            data[j] = data[j-1];
            j = j - 1;
        }

        data[j] = x;
    }

}

void writeSentinel(research* data){

    int key = 0;
    data[key].monthly_earnings = 0;
    data[key].number_of_people = 0;
    data[key].number_of_people_school_age = 0;
    strcpy(data[key].address, "SENTINEL");

}