#include <stdio.h>
#include <math.h>

void main(){

    for(;;){
        int n = 0, binary[100], binarybuffer[100], shiftedbinary[100], count = 0, resdec = 0;

        printf("\n\n\nNumber of seats: ");
        scanf("%d", &n);

        if(n <= 0){break;}

        while(n>0){
            int buffer = n%2;
            n = n/2;

            binarybuffer[count] = buffer;
            count++;
        }

        for(int j = 0; j <= count; j++) 
        {
            binary[count - j] = binarybuffer[j - 1];
        }

        shiftedbinary[count] = binary[0];
        for(int k = 0; k < count; k++) 
        {
            shiftedbinary[k] = binary[k];
        }
        shiftedbinary[0] = 0;

        for(int m = 0; m <= count; m++) {
            resdec = pow(2, count - m) * shiftedbinary[m] + resdec;
        }

        printf("Winning seat: %d", resdec);

    }

}