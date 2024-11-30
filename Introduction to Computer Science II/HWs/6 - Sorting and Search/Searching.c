//Atividade Pratica 6
//Nome:       Lucas Miranda
//Numero USP: 12542838
//Professor:  Renato Tinós

#include <stdio.h>
#include <string.h>
#include <math.h>

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
int search(research *data, int N, char* x);

void main(){

    research data[SIZE+1];
    FILE *fpointer;
    char x[ADDRESS_SIZE];
    int selection = 0, loop = 1, res = -1;

    //Read
    fpointer = fopen("data.txt","rt+");
    for(int i=0; i<SIZE; i++){
        char dummy;
        fgets(data[i].address, ADDRESS_SIZE, fpointer);
        fscanf(fpointer, "%d", &data[i].monthly_earnings);
        fscanf(fpointer, "%d", &data[i].number_of_people);
        fscanf(fpointer, "%d", &data[i].number_of_people_school_age);
        fscanf(fpointer, "\n\n", &dummy);
    }

    //Menu loop
    while(loop == 1){

        //Menu
        printf("\n===========================\n");
        printf("1 - List houses.\n");
        printf("2 - Search house.\n");
        printf("3 - Close.\n");
        printf("===========================\n");
        printf("Type option: ");
        scanf("%d", &selection);

        //List houses
        if(selection == 1){
            printf("\n===========================\n");
            printf("   (ADDRESS)\n");
            printf("   (MONTHLY EARNINGS - NUMBER OF PEOPLE - NUMBER OF SCHOOL AGE PEOPLE)\n");
            for(int i=0; i<SIZE; i++){
                printf("%d) ", i);
                for(int k = 0; data[i].address[k]; k++){
                    printf("%c", data[i].address[k]);
                }
                printf("   %d - %d - %d\n", data[i].monthly_earnings, data[i].number_of_people,  data[i].number_of_people_school_age);
            }
            printf("===========================\n");
        }

        //Search address
        if(selection == 2){
            printf("\n===========================\n");
                printf("Type search key: ");
                fflush(stdin);
                fgets(x, ADDRESS_SIZE, stdin);

                res = search(data, SIZE, x);

                if(res == -1){
                    printf("NOT FOUND! (Address must be exactly the same)");
                }else{
                    printf("\nFOUND! Address located at index %d.\n", res);
                    printf("\n(MONTHLY EARNINGS - NUMBER OF PEOPLE - NUMBER OF SCHOOL AGE PEOPLE)\n");
                    printf("%d - %d - %d\n", data[res].monthly_earnings, data[res].number_of_people,  data[res].number_of_people_school_age);
                }

                res = -1;
            printf("\n===========================\n");
        }

        //Close
        if(selection == 3){loop = 0;}

        fflush(stdin);
        selection = 0;
    }

}

//Binary Search
int search(research *data, int N, char* x){

    int L = 0;
    int R = N-1;
    int Success = 0;
    int m;

    while((L<=R)&&(Success == 0)){
        m = floor((R+L)/2);

        if(strcmp(data[m].address, x) == 0){ Success = 1; }
        else{
            if(strcmp(data[m].address, x) < 0){
                L = m + 1;
            }else{
                R = m-1;
            }
        }
    }

    if(Success == 0){
        return -1;
    }

    return m;
}

