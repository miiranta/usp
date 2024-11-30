#include <stdio.h>
#include <math.h>

//Algumas constantes
#define e 2.71828182846
#define pi 3.14159265359

//Prototipos
void calcGlobalVars();
double f(double x);
double simpson();
double probabilidade(double y);

//Vars
    //Intervalo de integração
    const double a = 0;
    const double b = 1.8;

    //Numero de pontos
    int n = 1001;

    //Outras
    double x[10000], h;


void main(){
    //Define os xn's e h
    calcGlobalVars();

    //Calcula a integral
    double valorIntegral = simpson();
    printf("Aproximacao da integral: %f\n", valorIntegral);

    //Calcula a probabilidade
    double valorProbabilidade = probabilidade(valorIntegral);
    printf("Probabilidade calculada: %f", valorProbabilidade);
}

void calcGlobalVars(){
    h = ( (b - a) / (n - 1) );
    
    for(int i = 0; i < n; i++){
        x[i] = a + h*i;
    }
}

double f(double x){

    //DEFINA A FUNÇÃO f AQUI!
    return pow(e, ( -(pow(x, 2) / 2) ));
}

double simpson(){

    double sum = 0;

    for(int i = 0; i < n; i++){
       
        //É um extremo
        if((i == 0) || (i == (n-1))){
            sum = f(x[i]) + sum;
            continue;
        }

        //É par
        if((i % 2) == 0){
            sum = f(x[i]) * 2 + sum;
            continue;
        }

        //É impar
        if((i % 2) != 0){
            sum = f(x[i]) * 4 + sum;
            continue;
        }

    }

    return ( sum * (h/3) ); 
}

double probabilidade(double y){
    return 0.5 + (1 / ( sqrt(2* pi) )) * y;
}