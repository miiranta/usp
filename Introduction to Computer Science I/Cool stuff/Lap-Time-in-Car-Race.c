#include <stdio.h>
#include <string.h>
#define max 250

int lapdata(int *laps, int n);

void main(){

    int n = 0, i = 0, data = 0;

    printf("How many laps (N)?\n");
    scanf("%d", &n);

    if(n < 0){return;};
    int laps[n];

    printf("\nSet time taken on each lap in seconds:\n");
    for(i = 0; i<n; i++){
        printf(" Lap %d: ", i);
        scanf("%d", &data);
        laps[i] = data;
    }

    lapdata(laps, n);

}

int lapdata(int *laps, int n){

    int i = 0, min = 99999999, minlapnumber = 0, sum = 0;
    float av = 0;

    for(i = 0;i<n;i++){
        if(laps[i] < min){
            min = laps[i];
            minlapnumber = i;
        }
        sum = laps[i] + sum;
    }
    printf("\nBest lap: %d\n", minlapnumber);
    printf("Fastest time: %d s\n", min);

    av = (float)sum/(float)n;
    printf("Average time on all laps: %f s", av);


}


    