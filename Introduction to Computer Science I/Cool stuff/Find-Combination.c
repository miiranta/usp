#include <stdio.h>

void main(){

    for(;;){

    int n = 2, k = 1;
    float nfat = 1, kfat = 1, nkfat = 1, soma;
    
        printf("\n\nValue of n: ");
        scanf("%d", &n);

        printf("Value of k: ");
        scanf("%d", &k);

    int nk = n - k;

        if(n > k){

            for(int i = 1; i<=n; i++){

                float nf = (float)i;
                nfat = nf*nfat;

            }
            printf("\nn Factorial: %f", nfat);

            for(int i = 1; i<=k; i++){

                float kf = (float)i;
                kfat = kf*kfat;

            }
            printf("\nk Factorial: %f", kfat);

            for(int i = 1; i<=nk; i++){

                float nkf = (float)i;
                nkfat = nkf*nkfat;

            }
            printf("\nn - k Factorial: %f", nkfat);

            float resposta = nfat / (kfat * nkfat);
            printf("\nResponse: %f", resposta);

        }else{
            printf("n must be smaller than k!\n");
        }

    }




}