#include <time.h>
#include <stdio.h>
#include <stdlib.h>


void main(){

srand(time(NULL));

int dummy;

//numero aleatorio
int aleatorio = rand() % 100 + 0;

//prints iniciais
printf("\n\n==========================================\n\n");
printf("Trabalho 1 Questao 2 - Professor Zhao - Lucas Miranda Mendonca Rezende");
printf("\n\n==========================================\n\n");

printf("\n\n\n\nGuessing Game\n");
printf("Tente acertar o numero de 0 a 100...\n\n");

//Variaveis
int selection, quantidade = 0;

//Loop de tentativas
do{

        //Loop de verificação do scanf
        for(;;){

            //Seleção
            printf("Seu palpite: ");
            scanf("%d", &selection);

            //Entre 0 e 100?
            if(selection > 100 || selection < 0){  
            printf("\n\n==========================================\n\n");
            printf("Invalido! Digite um numero entre 0 e 100");
            printf("\n\n==========================================\n\n");
            continue;
            }else{break;}

        }

        //Numero de palpites
        quantidade++;
        if(selection != aleatorio){
        printf("\n==========================================\n\n");
        printf("Voce fez %d palpites.\n", quantidade);
        }

        //numero é menor
        if(selection > aleatorio){
            printf("O numero e MENOR que o seu palpite!");
            printf("\n\n==========================================\n\n");
        }

        //numero é maior
        if(selection < aleatorio){
            printf("O numero e MAIOR que o seu palpite!");
            printf("\n\n==========================================\n\n");
        }

        //acertou
        if(selection == aleatorio){
            printf("\n==========================================\n\n");
            printf("PARABENS! Voce acertou.\n");
            printf("Foram necessarias %d tentativas.", quantidade);
            printf("\n\n==========================================\n\n");
        }
    

    //Bug fix
    fflush(stdin);


}while(selection != aleatorio);


//Buffer
printf("\n\nDigite qualquer numero + enter para fechar o programa...\n\n");
scanf("%d",&dummy);
fflush(stdin);



}