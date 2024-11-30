//Min O(1)
//Med O(n)
//Max O(n)


#include <stdio.h>
int linear_search(int* a, int N, int x);

void main(){
    
    //        i  0  1   2   3   4   5   6   7  8   9
    int a[10] = {9, 50, 40, 20, 30, 30, 10, 5, 12, 392};
    int N = 10;
    int x = 30;

    printf("%d", linear_search(a, N, x));

    //-1 => NOT FOUND

}


int linear_search(int* a, int N, int x){

    int i = 0;

    while(i<N && a[i] != x){
        i++;
    }

    if(i>=N){
        return -1;
    }

    return i;

}




