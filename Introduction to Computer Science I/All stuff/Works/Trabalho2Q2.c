#include <stdio.h>

//Lucas Miranda - Número USP: 12542838

void main(){

    //prints iniciais
    printf("\n\n==========================================\n\n");
    printf("Trabalho 2 Questao 2 - Professor Zhao - Lucas Miranda Mendonca Rezende");
    printf("\n\n==========================================\n\n");

    //vars
    int n = 1, m = 1, i = 0, data = 0, already = 0, amountOfIntersections = 0;

    //Get size of N
    printf("How many positions for vector N?\n");
    scanf("%d", &n);

    //Get size of M
    printf("\nHow many positions for vector M?\n");
    scanf("%d", &m);

    //Create vectors
    int vectorN[n], vectorM[m], intersection[m+n], *p;
    p = &intersection[0];

    //Set vector N
    printf("\nSet elements for vector N\n");
    for(i = 0; i<n; i++){
        printf(" Position %d: ", i);
        scanf("%d", &data);
        vectorN[i] = data;
    }

    //Set vector M
    printf("\nSet elements for vector M\n");
    for(i = 0; i<m; i++){
        printf(" Position %d: ", i);
        scanf("%d", &data);
        vectorM[i] = data;
    }

    //Test every combination
    for(int j = 0; j<n; j++){
    for(int k = 0; k<m; k++){

            //If matches
            if(vectorN[j]==vectorM[k]){

                
                //Test if it is already logged
                already = 0;
                for(int l = 0; l<amountOfIntersections; l++){

                    if(intersection[l] == vectorN[j]){
                    already = 1;
                    }

                }
                
                //If already logged, skip
                if(already == 1){
                    continue;
                }

                //If not logged, log
                amountOfIntersections++;
                *p = vectorN[j];
                p++;

            }

    }
    }

    //Print intersection
    printf("\n==========================================\n");
    printf("Intersection = { ");
    for(int z = 0; z < amountOfIntersections; z++){
        printf("%d ", intersection[z]);
    }
    printf("}");
    printf("\n==========================================\n");


}

