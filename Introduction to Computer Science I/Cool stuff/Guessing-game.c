#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void main(){

srand(time(NULL));

int dummy;
int aleatorio = rand() % 100 + 0;

printf("\n\n\n\nGuessing Game\n");
printf("Try to guess a number between 0 and 100...\n\n");

int selection, quantidade = 0;

do{

        for(;;){

            printf("Your guess: ");
            scanf("%d", &selection);

            if(selection > 100 || selection < 0){  
            printf("\n\n==========================================\n\n");
            printf("Invalid! Type a between 0 and 100...");
            printf("\n\n==========================================\n\n");
            continue;
            }else{break;}

        }

        quantidade++;
        if(selection != aleatorio){
        printf("\n==========================================\n\n");
        printf("You made %d guesses.\n", quantidade);
        }

        if(selection > aleatorio){
            printf("The number is SMALLER than your guess!");
            printf("\n\n==========================================\n\n");
        }

        if(selection < aleatorio){
            printf("The number is LARGER than your guess!");
            printf("\n\n==========================================\n\n");
        }

        if(selection == aleatorio){
            printf("\n==========================================\n\n");
            printf("CONGRATS! You got it.\n");
            printf("A total of %d guesses were made.", quantidade);
            printf("\n\n==========================================\n\n");
        }
    
    fflush(stdin);


}while(selection != aleatorio);


printf("\n\nType anything + enter to close...\n\n");
scanf("%d",&dummy);
fflush(stdin);



}