#include <stdio.h>

void main(){
    
    int n = 1, m = 1, i = 0, data = 0, already = 0, amountOfIntersections = 0;

    printf("How many positions for vector N?\n");
    scanf("%d", &n);

    printf("\nHow many positions for vector M?\n");
    scanf("%d", &m);

    int vectorN[n], vectorM[m], intersection[m+n], *p;
    p = &intersection[0];

    printf("\nSet elements for vector N\n");
    for(i = 0; i<n; i++){
        printf(" Position %d: ", i);
        scanf("%d", &data);
        vectorN[i] = data;
    }

    printf("\nSet elements for vector M\n");
    for(i = 0; i<m; i++){
        printf(" Position %d: ", i);
        scanf("%d", &data);
        vectorM[i] = data;
    }

    for(int j = 0; j<n; j++){
    for(int k = 0; k<m; k++){

            if(vectorN[j]==vectorM[k]){

                already = 0;
                for(int l = 0; l<amountOfIntersections; l++){

                    if(intersection[l] == vectorN[j]){
                    already = 1;
                    }

                }
                
                if(already == 1){
                    continue;
                }

                amountOfIntersections++;
                *p = vectorN[j];
                p++;

            }

    }
    }

    printf("\n==========================================\n");
    printf("Intersection = { ");
    for(int z = 0; z < amountOfIntersections; z++){
        printf("%d ", intersection[z]);
    }
    printf("}");
    printf("\n==========================================\n");


}

