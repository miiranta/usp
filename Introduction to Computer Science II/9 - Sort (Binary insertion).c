//Comparisons
    //All cases O(n.Log(2)n)
//Movements
    //Min O(n)
    //Med O(n^2)
    //Max O(n^2)


#include <stdio.h>
#include <math.h>
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
    int i, j, L, R, x, m;

    for(i = 2; i<=N; i++){

        x = a[i];
        
        //Binary quick search
        L = 1;
        R = i;

        while(L<R){
            
            m = floor((L+R)/2);

            if(a[m] <= x){
                L= m + 1;
            }else{
                R = m;
            }

        }

        //Movements
        j = i;

        while(j>R){
            a[j] = a[j-1];
            j--;
        }

        a[R] = x;


    }

}