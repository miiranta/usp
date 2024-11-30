#include <stdio.h>
#include <math.h>
#define MAX_ZEROS 1000
#define PI 3.14159265358979323846

//Vars
double zeros_intervals[MAX_ZEROS][2];
double zeros_final[MAX_ZEROS];
int zeros_intervals_i = 0;
int zeros_final_i = 0;

//Prototypes
void searchInterval();
void bissec(double an, double bn);
void sec(double an, double bn);
void regula_falsi(double an, double bn);

//Set your function
double f(double x){
    return PI * pow(x, 2) * ((9 - x) / 3) - 30;
}

//Config
double e = 0.00000001;
int interval_break_quantity = 10;
int NMAX = 1000;

//inicial search interval [a,b] a<b
double a = 0;
double b = 6;

//Main
void main(){

    //TVI
    searchInterval();

    //Found possible zero interval?
    for(int i = 0; i < zeros_intervals_i; i++){
        printf("Zero interval: [%f, %f]\n", zeros_intervals[i][0], zeros_intervals[i][1]);
        
        //Bissecção
        printf("\n  Testando metodo da bisseccao ============================\n\n");
        bissec(zeros_intervals[i][0], zeros_intervals[i][1]);
        printf("\n  =========================================================");

        //Secante
        printf("\n\n  Testando metodo da secante ==============================\n\n");
        sec(zeros_intervals[i][0], zeros_intervals[i][1]);
        printf("\n  =========================================================");

        //Regula-falsi
        printf("\n\n  Testando metodo da Regula-falsi =========================\n\n");
        regula_falsi(zeros_intervals[i][0], zeros_intervals[i][1]);
        printf("\n  =========================================================\n\n");

    }

    //Found "perfect" zeros?
    for(int i = 0; i < zeros_final_i; i++){
        printf("Zero found: %f\n", zeros_final[i]);
    }

}

void searchInterval(){
    double bs = fabs(b-a)/(double)interval_break_quantity;

    double t0 = a;
    double t1 = a + bs;

    do{
        
        if(f(t0) == 0){
            zeros_final[zeros_final_i] = t0;
            zeros_final_i++;
        }

        else if(f(t1) == 0){
            zeros_final[zeros_final_i] = t1;
            zeros_final_i++;
        }

        else if(f(t0)*f(t1) < 0){
            zeros_intervals[zeros_intervals_i][0] = t0;
            zeros_intervals[zeros_intervals_i][1] = t1;
            zeros_intervals_i++;

        }

        //printf("t0: %f  t1: %f \n", t0, t1);

        t0 = t1;
        t1 = t1 + bs;
    }while(t1 <= b);

    return;
}

void bissec(double an, double bn){
    double xn;
    int j = 0;

    do{
        xn = (an + bn) / 2;

        if(f(xn)*f(an) == 0){
            zeros_final[zeros_final_i] = xn;
            zeros_final_i++;
        }

        else if(f(xn)*f(an) > 0){
            an = xn;
        }

        else if(f(xn)*f(an) < 0){
            bn = xn;
        }

        printf("  %d - [%.20f , %.20f]\n", j, an, bn);
        j++;

    }while((fabs(an - bn) > e) && (j < NMAX));

    printf("  x* ~= %.20f\n", (an + bn)/2);
}

void sec(double an, double bn){
    double x0 = an;
    double x1 = bn;
    double x2;
    int j = 0;

    do{
        x2 = ( x0*f(x1) - x1*f(x0) ) / ( f(x1) - f(x0) );

        if((fabs(x2 - x1) / fabs(x2)) < e){
            break;
        }

        x0 = x1;
        x1 = x2;

        printf("  %d - %.20f\n", j, x2);
        j++;
        
    }while(j < NMAX);

    printf("  x* ~= %.20f\n", x2);
}

void regula_falsi(double an, double bn){
    double x0, x1, x2;
    int j = 0;

    do{
        x0 = an;
        x1 = bn;
        x2 = ( x0*f(x1) - x1*f(x0) ) / ( f(x1) - f(x0) );

        if( (fabs((x2 - an) / x2) < e) || (fabs((x2 - bn) / x2) < e) ){
            break;
        }

        if(f(x2)*f(an) > 0){
            an = x2;
        }
        else{
            bn = x2;
        }

        printf("  %d - [%.20f , %.20f]\n", j, an, bn);
        j++;

    }while(j < NMAX);

    printf("  x* ~= %.20f\n", x2);
}