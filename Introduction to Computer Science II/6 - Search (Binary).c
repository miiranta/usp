//Min O(1)
//Med O(Log(2)n)
//Max O(Log(2)n)


#include <stdio.h>
#include <math.h>
int search(int* a, int N, int x);

void main(){
    
    //        i  0  1  2  3   4   5   6   7   8   9
    int a[10] = {2, 4, 5, 34, 45, 53, 76, 87, 89, 90}; //NEEDS TO BE SORTED!! (Small) --> (Big)
    int N = 10;
    int x = 76;

    printf("%d", search(a, N, x));

    //-1 => NOT FOUND

}


int search(int* a, int N, int x){

    int L = 0;
    int R = N-1;
    int Success = 0;
    int m;

    while((L<=R)&&(Success == 0)){

        m = floor((R+L)/2);

        if(a[m]==x){ Success = 1; }
        else{

            if(a[m]<x){
                L = m + 1;
            }else{
                R = m-1;
            }

        }

    }

    if(Success == 0){
        return -1;
    }

    return m;

}