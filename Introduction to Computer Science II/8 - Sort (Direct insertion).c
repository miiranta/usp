//Comparisons
    //Min O(n)
    //Med O(n^2)
    //Max O(n^2)
//Movements
    //Min O(n)
    //Med O(n^2)
    //Max O(n^2)


#include <stdio.h>
void sort(int* a, int N);

void main(){
    //           s  1  2   3   4   5   6   7   8  9   10    (s is SENTINEL!)
    int a[11] = {1, 9, 50, 40, 20, 30, 30, 10, 5, 12, 392}; //Has to be declared N+1 to avoid problems with sentinel!
    int N = 10;

    sort(a, N);

    for(int i=1; i<=N; i++){   //NOTE: Starts at a[1] ends in a[N]  (a[0] is sentinel)
        printf("%d ", a[i]);
    }

}


void sort(int* a, int N){
    int i, j, x;

    for(i = 2; i<=N; i++){

        x = a[i];
        a[0] = x;
        j = i;

        //Linear search with sentinel
        while(x < a[j-1]){
            a[j] = a[j-1];
            j = j - 1;
        }

        a[j] = x;

    }

}