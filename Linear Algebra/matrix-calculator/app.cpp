#include <iostream>
#include "handler.hpp"
using namespace std;

//LOOP?
int loop = 1;

int main(){

    cout.precision(5);
    int close = 0, option = 0, Aset, Bset, buffer;

    Aset = setMatrixA();
    do{
        displayOptions();
        fflush(stdin);
        cin >> option;

        if(option == 0){close = 1;}

        if(option == 1){
            setMatrix();
            cin >> option;
            if(option == 1){Aset = setMatrixA();}
            if(option == 2 && Aset >= 1){Bset = setMatrixB();}
            continue;
        }

        if(option == 2 && Aset >= 1 && Bset >= 1){
            switchAandB();
            buffer = Aset;
            Aset = Bset;
            Bset = buffer;
            continue;
        }

        if(option == 3 && Aset >= 2){displayCofactorMatrix();}

        if(option == 4 && Aset >= 2){
            displayInverseMatrix();
            cin >> option;
            if(option == 1){displayInverseMatrixByCofactor();}
            if(option == 2){displayInverseMatrixByScaling();}
        }

        if(option == 5 && Aset >= 2){
            displayDeterminant();
            cin >> option;
            if(option == 1){displayDeterminantByPermutation();}
            if(option == 2){displayDeterminantByTriangle();}
        }

        if(option == 6 && Aset >= 2){displayTransposedMatrix();}

        if(option == 7 && Aset >= 1){
            displayMultiplyMatrixByK();
        }

        if(option == 8 &&  Aset >= 1 && Bset >= 1 && isPossibleToMultiply()){
            displayMultiplyMatrixByMatrix();
        }

        if(option == 9 && Aset >= 1 && Bset >= 1 && isEqualSize()){
            displaySumMatrix();
        }

        if(option == 10 && Aset >= 1 && Bset >= 1 && isEqualSize()){
            displaySubtractMatrix();
        }

        if(close == 0){sleeper();}

    }while(loop == 1 && close == 0);

}



