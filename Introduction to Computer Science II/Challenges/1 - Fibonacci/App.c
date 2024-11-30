#include <stdio.h>
#include <math.h>

//Presets
#define N0 0
#define N1 1
#define N2 1
#define ARRAYLIMIT 1000

int saveSeries[ARRAYLIMIT];
int recursive_LessEfficient(int n);
int recursive_MoreEfficient(int n);

//Main function
void main(){
    int res;

    //Define n
    //n0 n1 n2 n3 n4 n5
    //0  1  1  2  3  5
    int n = 10;

    //res = recursive_LessEfficient(n);
    //printf("%d \n", res);
    res = recursive_MoreEfficient(n);
    printf("%d", res);


}

int recursive_LessEfficient(int n){

    //First terms - Defines N0, N1 and N2
    if(n == N0 || n == N1){return n;}
    if(n == 2){return 1;}

    //Recursion - Makes recursion with 2 functions
    else{
        return recursive_LessEfficient(n-2) + recursive_LessEfficient(n-1);
    }

}

int recursive_MoreEfficient(int n){

    //First terms - Defines N0, N1 and N2
    if(n == N0 || n == N1){saveSeries[n] = n;}
    else if(n == 2){saveSeries[n] = 1; saveSeries[n-1] = 1;}

    //Recursion - Makes recursion with 1 function and saved results
    else{
        saveSeries[n] = recursive_MoreEfficient(n-1) + saveSeries[n-2];
    }

    return saveSeries[n];
}


