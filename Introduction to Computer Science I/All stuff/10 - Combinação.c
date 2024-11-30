#include <stdio.h>

void main(){

    for(;;){

    int n = 2, k = 1;
    float nfat = 1, kfat = 1, nkfat = 1, soma;
    
        printf("\n\nValor n: ");
        scanf("%d", &n);

        printf("Valor k: ");
        scanf("%d", &k);

    int nk = n - k;

        if(n > k){

            for(int i = 1; i<=n; i++){

                float nf = (float)i;
                nfat = nf*nfat;

            }
            printf("\nn fatorial: %f", nfat);

            for(int i = 1; i<=k; i++){

                float kf = (float)i;
                kfat = kf*kfat;

            }
            printf("\nk fatorial: %f", kfat);

            for(int i = 1; i<=nk; i++){

                float nkf = (float)i;
                nkfat = nkf*nkfat;

            }
            printf("\nn - k fatorial: %f", nkfat);

            float resposta = nfat / (kfat * nkfat);
            printf("\nResposta: %f", resposta);

        }else{
            printf("n deve ser menor que k!\n");
        }


    }




}