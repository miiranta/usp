//Code by Lucas Miranda - T3

#include <stdio.h>
#include <math.h>

void main(){

    for(;;){

        int n = 0, binary[100], binarybuffer[100], shiftedbinary[100], count = 0, resdec = 0;

        printf("\n\n\nDigite o numero de cadeiras: ");
        scanf("%d", &n);

        if(n <= 0){break;}

        //Converte para Binario
        while(n>0){

            int buffer = n%2;
            n = n/2;

            //Registra e avança o array
            binarybuffer[count] = buffer;
            count++;

        }

        //Inverte o array binary
        for(int j = 0; j <= count; j++) 
        {

            binary[count - j] = binarybuffer[j - 1];
            
        }

        //Print Binary
        printf("Numero em binario: ");
        for(int i = 0; i < count; i++) {printf("%d ", binary[i]);}

        //Transfere o primeiro termo para o final
        shiftedbinary[count] = binary[0];
        for(int k = 0; k < count; k++) 
        {

            shiftedbinary[k] = binary[k];
            
        }
        shiftedbinary[0] = 0;

        //Print ShiftedBinary
        printf("\nJogando o primeiro termo para o ultimo: ");
        for(int l = 0; l <= count; l++) {printf("%d ", shiftedbinary[l]);}


        //Converte para decimal
        for(int m = 0; m <= count; m++) {
            
            resdec = pow(2, count - m) * shiftedbinary[m] + resdec;
    
        }

        //Print resposta
        printf("\nO assento vencedor: %d", resdec);



    }

}