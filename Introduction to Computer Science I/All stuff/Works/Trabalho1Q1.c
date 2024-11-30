//Code by Lucas Miranda - T3

#include <stdio.h>
#include <math.h>

void main(){

    printf("\n\n==========================================\n\n");
    printf("Trabalho 1 Questao 1 - Professor Zhao - Lucas Miranda Mendonca Rezende");


    //Vars q não devem resetar com o loop
    float a = 0, b = 0, c = 0, d = 0, e = 0, max = 0, min = 999999, media, mediapon, soma, dp, var;
    float pa = 1, pb = 1, pc = 1, pd = 1, pe = 1, ptotal;
    float da, db, dc, dd, de;
    float vetor[100] = {0, 0, 0, 0, 0};
    int j;

    //For para o programa reiniciar sozinho
    for(;;){

        //Vars que resetam com o loop
        int selection = 0, dummy = 0;
       
        printf("\n\n==========================================\n\n");

        //UI do menu
        printf("Menu\n");
        printf("1. Selecionar os numeros\n");
        printf("2. Ver os numeros\n");
        printf("3. Media Aritmetica Simples\n");
        printf("4. Media Ponderada\n");
        printf("5. Desvio Padrao\n");
        printf("6. Maior Valor\n");
        printf("7. Menor Valor\n");

        //Loop de Seleção + Repeat se Invalido
        for(;;){

            //Seleção
            printf("Digite o numero correspondente: ");
            scanf("%d", &selection);

            //Numero Invalido
            if(selection == 0){
                printf("\nInvalido! Digite um numero...\n\n");
                fflush(stdin);
            }else{break;}

        }

        //Selecionar numeros
        if(selection == 1){

            printf("\n\n==========================================\n\n");

            printf("\nDigite o valor de A: ");
            scanf("%f", &a);
            vetor[0] = a;

            printf("\nDigite o valor de B: ");
            scanf("%f", &b);
            vetor[1] = b;

            printf("\nDigite o valor de C: ");
            scanf("%f", &c);
            vetor[2] = c;

            printf("\nDigite o valor de D: ");
            scanf("%f", &d);
            vetor[3] = d;

            printf("\nDigite o valor de E: ");
            scanf("%f", &e);
            vetor[4] = e;

        }

        //Ver os numeros
        if(selection == 2){

            printf("\n\n==========================================\n\n");

            printf("Valores:");
            printf("\nA: %f \nB: %f \nC: %f \nD: %f \nE: %f",a,b,c,d,e);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);


        }
         
        //Media Simples
        if(selection == 3){

            printf("\n\n==========================================\n\n");

            //Soma
            soma = a + b + c + d + e;

            printf("Soma: %f", soma);

            //Media
            media = soma / 5 ;

            printf("\nMedia Simples: %f/5 = %f", soma, media);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        //Media Ponderada
        if(selection == 4){

            printf("\n\n==========================================\n\n");

            //Pegar os pesos
            printf("A: %f", a);
            printf("\nDigite o peso de A: ");
            scanf("%f", &pa);

            printf("\n\nB: %f", b);
            printf("\nDigite o peso de B: ");
            scanf("%f", &pb);

            printf("\n\nC: %f", c);
            printf("\nDigite o peso de C: ");
            scanf("%f", &pc);

            printf("\n\nD: %f", d);
            printf("\nDigite o peso de D: ");
            scanf("%f", &pd);

            printf("\n\nE: %f", e);
            printf("\nDigite o peso de E: ");
            scanf("%f", &pe);

            //Media pon
            ptotal = pa + pb + pc + pd + pe;

            mediapon = (a*pa + b*pb + c*pc + d*pd + e*pe)/ptotal;

            printf("\n\nTotal dos pesos: %f\n", ptotal);
            printf("Media ponderada: %f*%f + %f*%f + %f*%f + %f*%f + %f*%f / %f = %f", a,pa,b,pb,c,pc,d,pd,e,pe,ptotal,mediapon);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        //Desvio Padrão
        if(selection == 5){

            printf("\n\n==========================================\n\n");

            //Soma
            soma = a + b + c + d + e;
            printf("Soma: %f", soma);

            //Media
            media = soma / 5 ;
            printf("\nMedia Simples: %f", media);

            //desvio
            da = a - media;
            db = b - media;
            dc = c - media;
            dd = d - media;
            de = e - media;
            printf("\n\nDesvio dos valores \nA: %f \nB: %f \nC: %f \nD: %f \nE: %f" , da, db, dc, dd, de);

            //variancia
            var = (pow(da, 2) + pow(db, 2) + pow(dc, 2) + pow(dd, 2) + pow(de, 2) )/5;
            printf("\n\nVariancia: %f", var);

            //Desvio padrão
            dp = sqrt(var);
            printf("\n\nDesvio padrao de populacao: %f", dp);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        //Maior Valor
        if(selection == 6){

            max = 0;

            for(j = 0; j<5; j++){
                if(vetor[j] > max){ max = vetor[j]; }
            }

            printf("\n\n==========================================\n\n");

            printf("Maior valor: %f", max);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        //Menor Valor
        if(selection == 7){

            min = 999999;

            for(j = 0; j<5; j++){
                if(vetor[j] < min){ min = vetor[j]; }
            }


            printf("\n\n==========================================\n\n");

            printf("Menor valor: %f", min);

            //Buffer
            printf("\n\nDigite qualquer numero + enter para voltar...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }
       
        

    }


}