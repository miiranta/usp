#include <stdio.h>
#include <math.h>
#define MAX_ZEROS 1000

//Vars
double zeros_intervals[MAX_ZEROS][2];
double zeros_final[MAX_ZEROS];
int zeros_intervals_i = 0;
int zeros_final_i = 0;

//Prototypes
void searchInterval();
void bissec(double an, double bn);

//inicial search interval [a,b] a<b
double a = 0;
double b = 10;

//Set your function
double f(double x){
    return cos(x) - x;
}

//Config
double e = 0.000001;
int interval_break_quantity = 10;
int NMAX = 1000;

//Main
void main(){

    //TVI
    searchInterval();

    for(int i = 0; i < zeros_final_i; i++){
        printf("Zero found: %f\n", zeros_final[i]);
    }

    for(int i = 0; i < zeros_intervals_i; i++){
        printf("Zero interval %d: t0 = %f  t1 = %f\n", i, zeros_intervals[i][0], zeros_intervals[i][1]);
        
        //Bissecção
        bissec(zeros_intervals[i][0], zeros_intervals[i][1]);
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

    printf("  x* ~= %.20f", (an + bn)/2);
}

