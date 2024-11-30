//Comparisons
    //All cases O(n^2)
//Movements
    //Min O(1)
    //Med O(n^2)
    //Max O(n^2)


#include <stdio.h>
void sort(int* a, int N);

void main(){
    //           s  1  2   3   4   5   6   7   8  9    10     (NO SENTINEL! s is ignored)
    int a[11] = {0, 9, 50, 40, 20, 30, 30, 10, 5, 392, 23};   //Has to be declared N+1
    int N = 10;

    sort(a, N);

    for(int i=1; i<=N; i++){  //NOTE: Starts at a[1] ends in a[N]  (a[0] is ignored)
        printf("%d ", a[i]);
    }

}


void sort(int* a, int N){
    int i, j, x;

    for(i=2; i<=N; i++){
        for(j=N; j>=i; j--){

            //Compare
            if(a[j-1] > a[j]){

                //Move
                x = a[j-1];
                a[j-1] = a[j];
                a[j] = x;
            }

        }
    }

}