#include <iostream>
#include <math.h>
#include <iomanip>
#include <algorithm>
#define MAXSIZE 100
#define MAXPERMUTS 100000
using namespace std;

int factorial(int n);
int makePermuts(int *permuts , int size);
double makeTransposeMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixAt[MAXSIZE][MAXSIZE]);
double makeCofactorMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixCo[MAXSIZE][MAXSIZE]);
double makeInverseMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixInv[MAXSIZE][MAXSIZE]);
double makeInverseMatrixByScaling(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixInv[MAXSIZE][MAXSIZE]);
double findCofactor(int print, int size, double matrixA[MAXSIZE][MAXSIZE], int line, int column);
double findDeterminantByTriangle(int print, int size, double matrixA[MAXSIZE][MAXSIZE]);
double findDeterminant(int print, int size, double matrixA[MAXSIZE][MAXSIZE]);
double printMatrix(int size, double matrixA[MAXSIZE][MAXSIZE], int how);
double getMatrix(int *sizePointer, double matrix[MAXSIZE][MAXSIZE], int *lines, int *columns);
double swapLine(int size, int line1, int line2, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]);
double swapColumn(int size, int column1, int column2, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]);
double multiplyLine(int size, int line, double k, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]);
double multiplyAndSumLines(int size, int line1, int line2, double k, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]);
double switchMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE]);
double multiplyMatrix(int linesA, int columnsA, double matrixA[MAXSIZE][MAXSIZE], double matrixMul[MAXSIZE][MAXSIZE]);
double multiplyMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixMul2[MAXSIZE][MAXSIZE]);
double sumMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixSum[MAXSIZE][MAXSIZE]);
double subtractMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixSub[MAXSIZE][MAXSIZE]);

int factorial(int n = 0){
    int res = 1; 
    if(n != 0){
        for(int i = 1; i<=n; i++){
            res = res*i;
        }
    }
    else if(n == 0){res = 1;}
    return res;
}

int makePermuts(int *permuts, int size){
    int i, j;
    int a[size+1];
    for(i = 0; i < size; i++){
        a[i] = i;
    }
    sort(a, a+size);
    j = 0;
    do {
        for(int i = 0; i<size; i++){
            permuts[i+j*size] = a[i];
        }
        j++;
    } while(next_permutation(a, a+size));
    return 0;
}

double makeTransposeMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixAt[MAXSIZE][MAXSIZE]){
    int i, j;
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            matrixAt[j][i] = matrixA[i][j];
        }
    }
    return 0;
}

double makeCofactorMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixCo[MAXSIZE][MAXSIZE]){
    double cofac;
    int i, j;
    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            if(print==1){cout << "\n";}
            cofac = findCofactor(print, size, matrixA, i, j);
            matrixCo[i][j] = cofac;
        }
    }
    return 0;
}

double makeInverseMatrix(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixInv[MAXSIZE][MAXSIZE]){

    double matrixCo[MAXSIZE][MAXSIZE], matrixCoT[MAXSIZE][MAXSIZE], determinant;
    int i,j;

    determinant = findDeterminant(0, size, matrixA);
    if(determinant == 0){
        cout << "There is no inverse!";
        return 0;
    }

    makeCofactorMatrix(0, size, matrixA, matrixCo);
    if(print == 1){
        cout << "Cofactor Matrix:";
        printMatrix(size, matrixCo, 0);
    }

    makeTransposeMatrix(0, size, matrixCo, matrixCoT);
    if(print == 1){
        cout << "\n\nAdjunct Matrix (cofactor transpose):";
        printMatrix(size, matrixCoT, 0);
    }

    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            matrixInv[i][j] = matrixCoT[i][j]/determinant;
        }
    }
    return 0;
}

double makeInverseMatrixByScaling(int print, int size, double matrixA[MAXSIZE][MAXSIZE], double matrixInv[MAXSIZE][MAXSIZE]){
    double k, determinant = 0;
    int i, j;

    determinant = findDeterminant(0, size, matrixA);
    if(determinant == 0){
        cout << "There is no inverse!\n";
        return 0;
    }

    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            matrixInv[i][j] = matrixA[i][j];
        }
    }

    for(i=0; i<size; i++){
        for(j=size; j<size*2; j++){
            matrixInv[i][j] = 0;
            matrixInv[i][i+size] = 1;
        }
    }

    for(i=0; i<size; i++){

            if(matrixInv[i][i] == 0){
                swapLine(size, i, i+1, matrixInv, matrixInv);
                if(print==1){
                    cout << "\nSwaping lines " << i+1 << " and " << i+2;
                    printMatrix(size, matrixInv, 1);
                    cout << endl;
                }
            }

            if(matrixInv[i][i] != 0){
                k = 1/matrixInv[i][i];
                multiplyLine(size, i, k, matrixInv, matrixInv);
                if(print==1){
                    cout << "\nMultiplying line " << i+1 << " by " << k;
                    printMatrix(size, matrixInv, 1);
                    cout << endl;
                }
            }

            for(j=0; j<size; j++){
                if(j != i){
                    if(matrixInv[j][i] != 0){
                        k = -matrixInv[j][i]/matrixInv[i][i];
                        multiplyAndSumLines(size, i, j, k, matrixInv, matrixInv);
                        if(print==1){
                            cout << "\nMultiplying line " << i+1 << " by " << k << " and adding to line " << j+1;
                            printMatrix(size, matrixInv, 1);
                            cout << endl;
                        }
                    }
                }
            }    
    }

    for(i=0; i<size; i++){
        for(j=0; j<size*2; j++){
            matrixInv[i][j] = matrixInv[i][j+size];
        }
    }

    return 0; 
}

double findCofactor(int print, int size, double matrixA[MAXSIZE][MAXSIZE], int line, int column){
    int i, j, jumpi, jumpj;
    double matrixCofDet[MAXSIZE][MAXSIZE], determinant, cofac;

    jumpi = 0;
    jumpj = 0;
    for(i = 0; i<size; i++){
        for(j = 0; j<size; j++){
            if(i != line){
                if(j != column){  
                    if(jumpj>=size-1){
                    jumpj = 0; 
                    jumpi++;
                    }
                    matrixCofDet[jumpj][jumpi] = matrixA[i][j];
                    jumpj++;
                }
            }
        }
    }

    if(print==1){
        if(print==1){cout << "---------------- " << "Cofactor a" << line+1 << column+1 << " ----------------";}
        for(i=0; i<size-1; i++){
            if(print==1){cout << endl;}
            for(j=0; j<size-1; j++){
                if(print==1){cout << " " << matrixCofDet[i][j];}
            }
        } 
        if(print==1){cout << "\n\n";}
    }

    if(print==1){cout << " Determinant: \n";}
    determinant = findDeterminant(print, size-1, matrixCofDet);

    cofac = pow((-1),line+column) * determinant;

    if(print==1){
        cout << "\n\n Calculating: \n";
        cout << " "<< determinant << " . (-1)^(" << line+1 <<"+"<< column+1 << ")";
        cout << "\n\n = " << cofac;
        cout << "\n----------------------------------------------";
    }

    return cofac;
}

double findDeterminantByTriangle(int print, int size, double matrixA[MAXSIZE][MAXSIZE]){
    double determinant = 1, bufferMatrix[MAXSIZE][MAXSIZE], k = 1;
    int i, j, swaps = 0;

    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            bufferMatrix[i][j] = matrixA[i][j];
        }
    }

    for(i=0; i<size-1; i++){

            if(bufferMatrix[i][i] == 0){

                swapLine(size, i, i+1, bufferMatrix, bufferMatrix);
                swaps++;
                if(print==1){
                    cout << "\nSwaping lines " << i+1 << " and " << i+2;
                    printMatrix(size, bufferMatrix, 0);
                    cout << endl;
                }

            }

            for(j=1; j<size-i; j++){
                if(bufferMatrix[i+j][i] != 0){
                    k = -bufferMatrix[i+j][i]/bufferMatrix[i][i];
                    multiplyAndSumLines(size, i, i+j, k, bufferMatrix, bufferMatrix);
                    if(print==1){
                        cout << "\nMultiplying line " << i+1 << " by " << k << " and adding to line " << i+j+1;
                        printMatrix(size, bufferMatrix, 0);
                        cout << endl;
                    }
                }
            }    
    }

    for(i=0; i<size; i++){
        determinant = bufferMatrix[i][i]*determinant;
    }

    if(print==1){
        cout << "\nMultiplying the main diagonal" << endl << determinant << endl;
        cout << "\nNumber of line swaps" << endl << swaps << endl;
    }

    determinant = pow((-1), swaps) * determinant;

    if(print==1){
        cout << "\nDeterminant" << endl << " = " << determinant << endl;
    }

    return determinant; 
}

double findDeterminant(int print, int size, double matrixA[MAXSIZE][MAXSIZE]){
    double determinant = 0, matrixElement = 1;
    int inv = 1, i, j, k, l, p;
    int permuts[MAXPERMUTS];
    int permutations = factorial(size);
    makePermuts(permuts, size);

    for(j=0; j<permutations; j++){
        inv = 1;
        for(k = 0; k < size; k++){
            for(l = 0; l < size-k; l++){
                if(permuts[j*size+k]>permuts[j*size+k+l]){
                    inv = -inv;
                }
            }
        }

        if(print == 1){
            if(inv == 1){
                cout << " + "; 
            }else{
                cout << " - "; 
            }
        }

        matrixElement = 1;
        for(i=0; i<size; i++){
            p = permuts[j*size+i];
            matrixElement = matrixElement*matrixA[i][p];
            if(print == 1) {cout << "a" << i+1 << p+1;}
        }
        if(print == 1) {cout << "      " << matrixElement*inv << endl;}
        determinant = matrixElement*inv + determinant;
    }

    if(print == 1) {cout << "\n = " << determinant;}
    return determinant;
}

double printMatrix(int size, double matrixA[MAXSIZE][MAXSIZE], int how){ 
    int i,j;
    if(how == 0){
        for(i=0; i<size; i++){
            cout << endl;
            for(j=0; j<size; j++){
                cout << setw(6) << matrixA[i][j] << " ";
            }
        }
    }

    if(how == 1){
        for(i=0; i<size; i++){
            cout << endl;
            for(j=0; j<size*2; j++){
                cout << setw(6) << matrixA[i][j] << " ";
            }
        }
    }

    return 0; 
}

double printMatrix(int lines, int columns, double matrixA[MAXSIZE][MAXSIZE], int how){ 
    int i,j;
    if(how == 0){
        for(i=0; i<columns; i++){
            cout << endl;
            for(j=0; j<lines; j++){
                cout << setw(6) << matrixA[i][j] << " ";
            }
        }
    }
    return 0;
}

double getMatrix(double matrix[MAXSIZE][MAXSIZE], int *lines, int *columns){
    int i, j, option;
    int msize = 0;
    for(;;){
        cout << "Select an option:" << endl;
        cout << "1 - Square Matrix" << endl;
        cout << "2 - Any size Matrix" << endl;

        fflush(stdin);
        cin >> option;

        if(option == 1){
            cout << "\nInsert matrix size (N x N): ";
            cin >> msize;
            cout << "\nInsert matrix elements (separated by space) and hit enter:\n";
            for(i=0;i<msize;i++){
                cout << " Line " << i+1 << ":\n ";
                for(j=0;j<msize;j++){
                    scanf("%lf", &matrix[i][j]);
                } 
            }
            *columns = msize;
            *lines = msize;
            return 0;
        }

        if(option == 2){
            cout << "\nInsert number of lines (I x j): ";
            cin >> *lines;
            cout << "\nInsert number of columns (i x J): ";
            cin >> *columns;
            cout << "\nInsert matrix elements (separated by space) and hit enter:\n";
            for(i=0;i<*lines;i++){
                cout << " Line " << i+1 << ":\n ";
                for(j=0;j<*columns;j++){
                    scanf("%lf", &matrix[i][j]);
                }
            }
            return 0;
        }
    }
}

double swapLine(int size, int line1, int line2, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]){
    int i, j;
    double buffer[size];

    for(i=0; i<size; i++){
        for(j=0; j<size*2; j++){
            matrixRes[j][i] = matrixA[j][i];
        }
    }
    
    for(i=0; i<size*2; i++){
        buffer[size] = matrixA[line1][i];
        matrixRes[line1][i] = matrixA[line2][i];
        matrixRes[line2][i] = buffer[size];

    }

    return 0;
}

double swapColumn(int size, int column1, int column2, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]){
    int i, j;
    double buffer[size];

    for(i=0; i<size; i++){
        for(j=0; j<size; j++){
            matrixRes[j][i] = matrixA[j][i];
        }
    }
    
    for(i=0; i<size; i++){
        buffer[size] = matrixA[i][column1];
        matrixRes[i][column1] = matrixA[i][column2];
        matrixRes[i][column2] = buffer[size];
    }

    return 0;
}

double multiplyLine(int size, int line, double k, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]){
    int i, j;

    for(i=0; i<size*2; i++){
        for(j=0; j<size; j++){
            matrixRes[j][i] = matrixA[j][i];
        }
    }
    
    for(i=0; i<size*2; i++){
        matrixRes[line][i] = matrixA[line][i]*k;
    }

    return 0;
}

double multiplyAndSumLines(int size, int line1, int line2, double k, double matrixA[MAXSIZE][MAXSIZE], double matrixRes[MAXSIZE][MAXSIZE]){
    double buffer[size*2];
    int i,j;

    for(i=0; i<size; i++){
        for(j=0; j<size*2; j++){
            matrixRes[j][i] = matrixA[j][i];
        }
    }

    for(i=0; i<size*2; i++){
        buffer[i] = matrixA[line1][i]*k;
    }

    for(i=0; i<size*2; i++){
        matrixRes[line2][i] = buffer[i] + matrixA[line2][i];
    }

    return 0;
}

double switchMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE]){
    double buffer[MAXSIZE][MAXSIZE];
    int i, j;

    for(i=0; i<linesA; i++){
        for(j=0; j<columnsA; j++){
            buffer[j][i] = matrixA[j][i];
        }
    }

    for(i=0; i<linesB; i++){
        for(j=0; j<columnsB; j++){
            matrixA[j][i] = matrixB[j][i];
        }
    }

    for(i=0; i<linesA; i++){
        for(j=0; j<columnsA; j++){
            matrixB[j][i] = buffer[j][i];
        }
    }

    return 0;
}

double multiplyMatrix(int linesA, int columnsA, double matrixA[MAXSIZE][MAXSIZE], double matrixMul[MAXSIZE][MAXSIZE]){
    double k;
    int i, j;
    cout << "Insert number to multiply matrix: ";
    cin >> k;

    for(i=0; i<linesA; i++){
        for(j=0; j<columnsA; j++){
            matrixMul[j][i] = matrixA[j][i]*k;
        }
    }

    return 0;
}

double multiplyMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixMul2[MAXSIZE][MAXSIZE]){
    int columnsMul2 = columnsB, linesMul2 = linesA, i, j, l, buffer;

    if(columnsA =! linesB){
        cout << "Impossible to multiply! ";
        return 0;
    }

    for(i=0; i<linesMul2; i++){
        for(j=0; j<columnsMul2; j++){
            buffer = 0;
            for(l=0; l<linesB; l++){
                buffer = matrixA[i][l]*matrixB[l][j] + buffer;
            }
            matrixMul2[i][j] = buffer;
        }
    }
    return 0;
}

double sumMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixSum[MAXSIZE][MAXSIZE]){
    int i, j;

    if((linesA =! linesB) || (columnsA =! columnsB)){
        cout << "Impossible to sum! ";
        return 0;
    }

    for(i=0; i<linesB; i++){
        for(j=0; j<columnsB; j++){
            matrixSum[j][i] = matrixA[j][i]+matrixB[j][i];
        }
    }
    return 0;
}

double subtractMatrices(int linesA, int columnsA, int linesB, int columnsB, double matrixA[MAXSIZE][MAXSIZE], double matrixB[MAXSIZE][MAXSIZE], double matrixSub[MAXSIZE][MAXSIZE]){
    int i, j;

    if((linesA =! linesB) || (columnsA =! columnsB)){
        cout << "Impossible to sum! ";
        return 0;
    }

    for(i=0; i<linesB; i++){
        for(j=0; j<columnsB; j++){
            matrixSub[j][i] = matrixA[j][i]-matrixB[j][i];
        }
    }
    return 0;
}




