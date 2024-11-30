//Comparisons
    //All cases O(n^2)
//Movements
    //All cases O(n)


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
    int i, j, x, i_smallest;

    for(i=1; i<=(N-1); i++){

        //Index of smallest element
        i_smallest = i;
        for(j=i+1; N>=j; j++){

            if(a[j] < a[i_smallest]){
                i_smallest = j;
            }

        }

        //Movements
        x = a[i];
        a[i] = a[i_smallest];
        a[i_smallest] = x;

    }

}


