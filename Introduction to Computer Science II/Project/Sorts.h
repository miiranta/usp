#include <stdio.h>
#include <string.h>
#include <math.h>

void sortDirectInsertion(Month* a, int N);
void sortBinaryInsertion(Month* a, int N);
void sortSelection(Month* a, int N);
void sortBubble(Month* a, int N);
void sortShaker(Month* a, int N);
void sortFusion(Month* a, int N);
void mpass(Month* a, int N, int p, Month* c);
void merge(Month* a, int L, int h, int R, Month* c);
void sortHeap(Month* a, int N);
void heapify(Month* a, int L, int R);
void sortQuick(Month* a, int N);
void qsort(Month* a, int L, int R);

//Global vars and counting
    int movements = 0;
    int comparisons = 0;

    void resetCount(){
        movements = 0;
        comparisons = 0;
    }

    void showCount(char* title){
        printf("== %s ==\n", title);
        printf("Movements: %d\nComparisons: %d\n\n", movements, comparisons);

        //Just to keep main() organized
        resetCount();
    }


//Direct Insertion
    void sortDirectInsertion(Month* a, int N){
        int i, j;
        Month x;

        for(i = 2; i<=N; i++){

            movements = movements + 2;
            x = a[i]; //m
            a[0] = x; //m
            j = i;
            
            comparisons++;
            while(strcmp(x.code, a[j-1].code)<0){ //c
            comparisons++;
            
                movements++;
                a[j] = a[j-1]; //m
                j = j - 1;
            }

            movements++;
            a[j] = x; //m

        }

    }


//Binary Insertion
    void sortBinaryInsertion(Month* a, int N){
        int i, j, L, R, m;
        Month x;

        for(i = 2; i<=N; i++){
            movements++;
            x = a[i]; //m

            L = 1;
            R = i;

            while(L<R){ 
                m = floor((L+R)/2);

                comparisons++;
                if(strcmp(x.code, a[m].code)>=0){ //c
                    L = m + 1;
                }else{
                    R = m;
                }
            }

            j = i;

            while(j>R){ 
                movements++;
                a[j] = a[j-1]; //m
                j--;
            }

            movements++;
            a[R] = x; //m
        }
    }


//Selection
    void sortSelection(Month* a, int N){
        int i, j, i_smallest;
        Month x;

        for(i=1; i<=(N-1); i++){

            i_smallest = i;
            for(j=i+1; N>=j; j++){

                comparisons++;
                if(strcmp(a[j].code, a[i_smallest].code)<0){ //c
                    i_smallest = j;
                }

            }

            movements = movements + 3;
            x = a[i]; //m
            a[i] = a[i_smallest]; //m
            a[i_smallest] = x; //m

        }

    }


//Bubble
    void sortBubble(Month* a, int N){
        int i, j;
        Month x;

        for(i=2; i<=N; i++){
            for(j=N; j>=i; j--){

                comparisons++;
                if(strcmp(a[j-1].code, a[j].code)>0){ //c

                    movements = movements + 3;
                    x = a[j-1]; //m
                    a[j-1] = a[j]; //m
                    a[j] = x; //m
                }

            }
        }
    }


//Shaker
    void sortShaker(Month* a, int N){
        int L=2, R=N, k=N;
        int j;
        Month x;

        do{
            for(j=R; j>=L; j=j-1){ 
                comparisons++;
                if(strcmp(a[j-1].code, a[j].code)>0){ //c
                    movements = movements + 3;
                    x = a[j-1]; //m
                    a[j-1] = a[j]; //m
                    a[j] = x; //m
                    k = j;
                }
            }

            L = k + 1;

            for(j=L; j<=R; j=j+1){
                comparisons++;
                if(strcmp(a[j-1].code, a[j].code)>0){ //c
                    movements = movements + 3;
                    x = a[j-1]; //m
                    a[j-1] = a[j]; //m
                    a[j] = x; //m
                    k = j;
                }
            }

            R = k - 1;
        }while(L<=R);

    }


//Fusion
    void sortFusion(Month* a, int N){
        int p = 1;
        Month c[N+1];

        while(p<N){
            mpass(a, N, p, c);
            p = 2*p;
            mpass(c, N, p, a);
            p = 2*p;
        }
    }

    void mpass(Month* a, int N, int p, Month* c){
        int i = 1, j;

        while(i<=N-2*p+1){
            merge(a, i, i+p-1, i+2*p-1, c);
            i = i + 2*p;
        }

        if(i+p-1<N){
            merge(a, i, i+p-1, N, c);
        }else{
            for(j=i; j<=N; j++){
                movements++;
                c[j] = a[j]; //m
            }
        }
    }

    void merge(Month* a, int L, int h, int R, Month* c){
        int i=L, j=h+1, k=L-1;

        while((i<=h)&&(j<=R)){ 
            k = k + 1;
            comparisons++;
            if(strcmp(a[i].code, a[j].code)<0){ //c
                movements++;
                c[k] = a[i]; //m
                i = i + 1;
            }else{
                movements++;
                c[k] = a[j]; //m
                j = j + 1;
            }
        }

        while(i<=h){
            k = k + 1;
            movements++;
            c[k] = a[i]; //m
            i = i + 1;
        }

        while(j<=R){
            k = k + 1;
            movements++;
            c[k] = a[j]; //m
            j = j + 1;
        }
    }


//Heap
    void sortHeap(Month* a, int N){
        int L, R;
        Month w;
        
        for(L=N/2; L>=1; L=L-1){
            heapify(a, L, N);
        }

        for(R=N; R>=2; R=R-1){
            movements = movements + 3;
            w = a[1]; //m
            a[1] = a[R]; //m
            a[R] = w; //m
            heapify(a, 1, R-1);
        }
    }

    void heapify(Month* a, int L, int R){
        int i=L, j=2*L, loop = 1;
        movements++;
        Month x=a[L]; //m

        if(j<R){ //c  (a little bit different, but works the same)
            comparisons++;
            if(strcmp(a[j].code, a[j+1].code)<0){
                j = j + 1;
            }
        }

        while((j<=R)&&(loop)){
            loop = 0;
            comparisons++;
            if(strcmp(x.code, a[j].code)<0){ //c  
                loop = 1;
                movements++;
                a[i] = a[j]; //m
                i = j;
                j = 2*j;
                if(j<R){
                    comparisons++;
                    if(strcmp(a[j].code, a[j+1].code)<0){ //c
                       j = j + 1; 
                    }
                }
            }
        }

        movements++;
        a[i] = x; //m
    }


//Quick
    void sortQuick(Month* a, int N){
        qsort(a, 1, N);
    }

    void qsort(Month* a, int L, int R){
        int i=L, j=R;
        int fl = floor((L+R)/2);
        movements++;
        Month x=a[fl], w;  //m

        do{
            comparisons++;
            while(strcmp(a[i].code, x.code)<0){ //c
            comparisons++;
                i = i + 1;
            }
            comparisons++;
            while(strcmp(x.code, a[j].code)<0){ //c
            comparisons++;
                j = j - 1;
            }

            if(i<=j){
                movements = movements + 3;
                w = a[i]; //m
                a[i] = a[j]; //m
                a[j] = w; //m
                i = i + 1;
                j = j - 1;
            }

        }while(i<=j);

        if(L<j){
            qsort(a, L, j);
        }

        if(i<R){
            qsort(a, i, R);
        }
    }