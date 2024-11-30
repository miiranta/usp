#include <stdio.h>
#include <math.h>

void main(){

    float a = 0, b = 0, c = 0, d = 0, e = 0, max = 0, min = 999999, media, mediapon, soma, dp, var;
    float pa = 1, pb = 1, pc = 1, pd = 1, pe = 1, ptotal;
    float da, db, dc, dd, de;
    float vetor[100] = {0, 0, 0, 0, 0};
    int j;

    for(;;){
        int selection = 0, dummy = 0;
       
        printf("\n\n==========================================\n\n");

        printf("Menu\n");
        printf("1. Select numbers\n");
        printf("2. See numbers\n");
        printf("3. Average\n");
        printf("4. Weighted average\n");
        printf("5. Standard deviation\n");
        printf("6. Biggest\n");
        printf("7. Smallest\n");

        for(;;){

            printf("Type number: ");
            scanf("%d", &selection);

            if(selection == 0){
                printf("\nInvalid! Type another number...\n\n");
                fflush(stdin);
            }else{break;}

        }

        if(selection == 1){

            printf("\n\n==========================================\n\n");

            printf("\nType A: ");
            scanf("%f", &a);
            vetor[0] = a;

            printf("\nType B: ");
            scanf("%f", &b);
            vetor[1] = b;

            printf("\nType C: ");
            scanf("%f", &c);
            vetor[2] = c;

            printf("\nType D: ");
            scanf("%f", &d);
            vetor[3] = d;

            printf("\nType E: ");
            scanf("%f", &e);
            vetor[4] = e;

        }

        if(selection == 2){

            printf("\n\n==========================================\n\n");

            printf("Values:");
            printf("\nA: %f \nB: %f \nC: %f \nD: %f \nE: %f",a,b,c,d,e);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);


        }

        if(selection == 3){

            printf("\n\n==========================================\n\n");

            soma = a + b + c + d + e;

            printf("Sum: %f", soma);

            media = soma / 5 ;

            printf("\nAverage: %f/5 = %f", soma, media);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        if(selection == 4){

            printf("\n\n==========================================\n\n");

            printf("A: %f", a);
            printf("\nWeight of A: ");
            scanf("%f", &pa);

            printf("\n\nB: %f", b);
            printf("\nWeight of B: ");
            scanf("%f", &pb);

            printf("\n\nC: %f", c);
            printf("\nWeight of C: ");
            scanf("%f", &pc);

            printf("\n\nD: %f", d);
            printf("\nWeight of D: ");
            scanf("%f", &pd);

            printf("\n\nE: %f", e);
            printf("\nWeight of E: ");
            scanf("%f", &pe);

            //Media pon
            ptotal = pa + pb + pc + pd + pe;

            mediapon = (a*pa + b*pb + c*pc + d*pd + e*pe)/ptotal;

            printf("\n\nWeight total: %f\n", ptotal);
            printf("Weighted average: %f*%f + %f*%f + %f*%f + %f*%f + %f*%f / %f = %f", a,pa,b,pb,c,pc,d,pd,e,pe,ptotal,mediapon);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        if(selection == 5){

            printf("\n\n==========================================\n\n");

            soma = a + b + c + d + e;
            printf("Sum: %f", soma);

            media = soma / 5 ;
            printf("\nAverage: %f", media);

            da = a - media;
            db = b - media;
            dc = c - media;
            dd = d - media;
            de = e - media;
            printf("\n\nDeviantion \nA: %f \nB: %f \nC: %f \nD: %f \nE: %f" , da, db, dc, dd, de);

            var = (pow(da, 2) + pow(db, 2) + pow(dc, 2) + pow(dd, 2) + pow(de, 2) )/5;
            printf("\n\nVariance: %f", var);

            dp = sqrt(var);
            printf("\n\nStandard deviation of population: %f", dp);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        if(selection == 6){

            max = 0;

            for(j = 0; j<5; j++){
                if(vetor[j] > max){ max = vetor[j]; }
            }

            printf("\n\n==========================================\n\n");

            printf("Biggest value: %f", max);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }

        if(selection == 7){

            min = 999999;

            for(j = 0; j<5; j++){
                if(vetor[j] < min){ min = vetor[j]; }
            }

            printf("\n\n==========================================\n\n");

            printf("Smallest value: %f", min);

            printf("\n\nType any number + enter to return...\n\n");
            scanf("%d",&dummy);
            fflush(stdin);

        }
       
        

    }


}