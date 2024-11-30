#include <iostream>
#include "functions.hpp"
#define MAXSIZE 100
using namespace std;

int i, j;
int Aset, Bset, linesA, linesB, columnsA, columnsB, dummy, buffer, k;
double matrixA[MAXSIZE][MAXSIZE], matrixB[MAXSIZE][MAXSIZE], matrixAt[MAXSIZE][MAXSIZE], matrixCo[MAXSIZE][MAXSIZE], matrixInv[MAXSIZE][MAXSIZE], matrixBuffer[MAXSIZE][MAXSIZE], matrixMul[MAXSIZE][MAXSIZE], matrixMul2[MAXSIZE][MAXSIZE], matrixSum[MAXSIZE][MAXSIZE], matrixSub[MAXSIZE][MAXSIZE], determinant; 

void setMatrix();
int setMatrixA();
int setMatrixB();
void displayOptions();
void displayDeterminant();
void displayDeterminantByPermutation();
void displayDeterminantByTriangle();
void displayTransposedMatrix();
void displayCofactorMatrix();
void displayInverseMatrix();
void displayInverseMatrixByCofactor();
void displayInverseMatrixByScaling();
void sleeper();
int isEqualSize();
int isPossibleToMultiply();
void switchAandB();
void displayMultiplyMatrixByK();
void displayMultiplyMatrixByMatrix();
void displaySumMatrix();
void displaySubtractMatrix();


void setMatrix(){
    cout << "\n================================\n\n";
        cout << "Select an option:" << endl;
        cout << "1 - Matrix A" << endl;
        if(Aset >= 1){cout << "2 - Matrix B" << endl;}
        cout << "0 - Return" << endl;
    cout << "\n================================\n\n";
}

int setMatrixA(){
    cout << "\n================================\n\n";
        getMatrix(matrixA, &columnsA, &linesA);
    cout << "\n================================\n\n";
    if(columnsA == linesA){Aset = 2;}else{Aset = 1;}
    return Aset;
}

int setMatrixB(){
    cout << "\n================================\n\n";
        getMatrix(matrixB, &columnsB, &linesB);
    cout << "\n================================\n\n";
    if(columnsB == linesB){Bset = 2;}else{Bset = 1;}
    return Bset;
}

void displayOptions(){
    cout << "\n================================\n\n";
        if(Aset >= 1){
            cout << "Matrix A:";
            printMatrix(linesA, columnsA, matrixA, 0);
            cout << "\n\n";
        }
        if(Bset >= 1){
            cout << "Matrix B:";
            printMatrix(linesB, columnsB, matrixB, 0);
            cout << "\n\n";
        }
        cout << "Select an option:" << endl;
        cout << "1 - Set Matrices" << endl;
        if(Aset >= 1 && Bset >= 1){
            cout << "2 - Switch matrix A and B " << endl;
        }
        if(Aset >= 2){
            cout << "3 - Matrix A: Cofactor " << endl;
            cout << "4 - Matrix A: Inverse" << endl;
            cout << "5 - Matrix A: Determinant" << endl;
            cout << "6 - Matrix A: Transpose " << endl;
        }
        if(Aset >= 1 ){
            cout << "7 - Matrix A: Multiply by k " << endl;
        }
        if(Aset >= 1 && Bset >= 1 && isPossibleToMultiply()){
            cout << "8 - Multiply Matrix A and B " << endl;
        }
        if(Aset >= 1 && Bset >= 1 && isEqualSize()){
            cout << "9 - Sum Matrix A and B (A+B)" << endl;
            cout << "10 - Subtract Matrix A and B (A-B)" << endl;
        }
        cout << "0 - Close" << endl;
    cout << "\n================================\n\n";
}

void displayDeterminant(){
    cout << "\n================================\n\n";
        cout << "Select an option:" << endl;
        cout << "1 - Using permutation" << endl;
        cout << "2 - Using scaling (triangle matrix)" << endl;
        cout << "0 - Return" << endl;
    cout << "\n================================\n\n";
}

void displayDeterminantByPermutation(){
    cout << "\n================================\n\n";
        cout << "Determinant by permutation: \n";
        determinant = findDeterminant(1, linesA, matrixA);
    cout << "\n\n================================\n\n";
}

void displayDeterminantByTriangle(){
    cout << "\n================================\n\n";
        cout << "Determinant by scaling: \n";
        findDeterminantByTriangle(1, linesA, matrixA);
    cout << "\n================================\n\n";
}

void displayTransposedMatrix(){
    cout << "\n================================\n\n";
        cout << "Matrix A Transpose:";
        makeTransposeMatrix(1, linesA, matrixA, matrixAt);
        printMatrix(linesA, matrixAt, 0);
    cout << "\n\n================================\n\n";
}

void displayCofactorMatrix(){
    cout << "\n\n================================\n\n";
        cout << "Calculating Cofactors: \n";
        makeCofactorMatrix(1, linesA, matrixA, matrixCo);
        cout << "\n\nCofactor Matrix:";
        printMatrix(linesA, matrixCo, 0);
    cout << "\n\n================================\n\n";
}

void displayInverseMatrix(){
    cout << "\n================================\n\n";
        cout << "Select an option:" << endl;
        cout << "1 - Using cofactors" << endl;
        cout << "2 - Using scaling" << endl;
        cout << "0 - Return" << endl;
    cout << "\n================================\n\n";
}

void displayInverseMatrixByCofactor(){
    cout << "\n\n================================\n\n";
        makeInverseMatrix(1, linesA, matrixA, matrixInv);
        cout << "\n\nInverse Matrix: ";
        printMatrix(linesA, matrixInv, 0);
    cout << "\n\n================================\n\n";
}

void displayInverseMatrixByScaling(){
    cout << "\n================================\n\n";
        makeInverseMatrixByScaling(1, linesA, matrixA, matrixInv);
        cout << "\nInverse Matrix: ";
        printMatrix(linesA, matrixInv, 0);
    cout << "\n\n================================\n\n";
}

void sleeper(){
    cout << "Press 0 + enter to return..." << endl;
    cin >> dummy;
}

int isEqualSize(){
    if(linesA == linesB && columnsA == columnsB){return 1;}
    return 0;
}

int isPossibleToMultiply(){
    if(columnsA == linesB){return 1;}
    return 0;
}

void switchAandB(){
    switchMatrices(linesA, columnsA, linesB, columnsB, matrixA, matrixB);
    buffer = linesA;
    linesA = linesB;
    linesB = buffer;

    buffer = columnsA;
    columnsA = columnsB;
    columnsB = buffer;

    buffer = Aset;
    Aset = Bset;
    Bset = buffer;
}

void displayMultiplyMatrixByK(){
    cout << "\n================================\n\n";
        multiplyMatrix(linesA, columnsA, matrixA, matrixMul);
        cout << endl << "Matrix A multiplied by k:" ;
        printMatrix(linesA, columnsA, matrixMul, 0);
        cout << endl;
    cout << "\n================================\n\n";
}

void displayMultiplyMatrixByMatrix(){
    cout << "\n================================\n\n";
        multiplyMatrices(columnsA, linesA, columnsB, linesB, matrixA, matrixB, matrixMul2);
        cout << "Matrix A multiplied by B:";
        printMatrix(columnsA, linesB, matrixMul2, 0);
        cout << endl;
    cout << "\n================================\n\n";
}

void displaySumMatrix(){
    cout << "\n================================\n\n";
        sumMatrices(linesA, columnsA, linesB, columnsB, matrixA, matrixB, matrixSum);
        cout << "Matrix A + B:";
        printMatrix(linesA, columnsA, matrixSum, 0);
        cout << endl;
    cout << "\n================================\n\n";
}

void displaySubtractMatrix(){
    cout << "\n================================\n\n";
        subtractMatrices(linesA, columnsA, linesB, columnsB, matrixA, matrixB, matrixSub);
        cout << "Matrix A - B:";
        printMatrix(linesA, columnsA, matrixSub, 0);
        cout << endl;
    cout << "\n================================\n\n";
}

