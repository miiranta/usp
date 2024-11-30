#include <iostream>
#include <math.h>
#include <cmath>
#include <array>
#include <iomanip>
#include <algorithm>
#include <vector>
#define MAXSIZE 100
using namespace std;

const double EulerConstant = std::exp(1.0);

void printArray(double array[MAXSIZE], int size, int how = 0);
double expectedRisk(double values[MAXSIZE], double chances[MAXSIZE], int size);
double expectedReturn(double values[MAXSIZE], double chances[MAXSIZE], int size);
double discreteReturn(double priceNow, double priceBefore);
double discreteReturn(double prices[MAXSIZE], double response[MAXSIZE], int size);
double continousReturn(double priceNow, double priceBefore);
double continousReturn(double prices[MAXSIZE], double response[MAXSIZE], int size);
double avarageDiscreteReturn(double prices[MAXSIZE], int size);
double avarageContinousReturn(double prices[MAXSIZE], int size);
double avarageReturnWithPercentage(double percentage[MAXSIZE], int size);
double getPossibleDiscreteReturnsWithPrices(double priceNow ,double prices[MAXSIZE], double response[MAXSIZE], int size);
double getPossibleContinousReturnsWithPrices(double priceNow ,double prices[MAXSIZE], double response[MAXSIZE], int size);
double getRisk(double prices[MAXSIZE], int size);
double getRiskWithPercentage(double percentage[MAXSIZE], int size);
double getVolatility(double prices[MAXSIZE], int size);
double getRiskPerRuturn(double prices[MAXSIZE], int size);
double getRiskPerRuturnWithPercentage(double percentage[MAXSIZE], int size);
double getVolatilityPerRuturn(double prices[MAXSIZE], int size);
double getAvarage(double values[MAXSIZE], int size);
double getAvarageWithWeight(double values[MAXSIZE], double weight[MAXSIZE], int size);
double getMedian(double values[MAXSIZE], int size);
double getMode(double array[MAXSIZE], int size);
double getVariance(double array[MAXSIZE], int size);
double getVarianceWithWeight(double array[MAXSIZE], double weight[MAXSIZE], int size);
double getStandardDeviation(double array[MAXSIZE], int size);
void sortArray(double values[MAXSIZE], double sorted[MAXSIZE], int size);

void printArray(double array[MAXSIZE], int size, int how){
    int i = 0;

    if(how == 0){
        for(int i = 0; i<size; i++){
            cout << setw(7) << array[i] << " ";
        }
        cout << endl; 
    }

    //Percentage
    if(how == 1){
        for(int i = 0; i<size; i++){
            cout << setw(6) << array[i]*100 << "% ";
        }
        cout << endl; 
    }

    //Percentage No multiply
    if(how == 2){
        for(int i = 0; i<size; i++){
            cout << setw(6) << array[i] << "% ";
        }
        cout << endl; 
    }
    
}

double expectedRisk(double values[MAXSIZE], double chances[MAXSIZE], int size){
    double risk = getVarianceWithWeight(values, chances, size);
    return pow(risk, 0.5);
}

double expectedReturn(double values[MAXSIZE], double chances[MAXSIZE], int size){
    int i = 0;
    double res = 0;
    double chancesSum = 0;
   
    for(int i = 0; i < size; i++){
        res = values[i] * chances[i] + res;
        chancesSum = chancesSum + chances[i];
    }

    if(chancesSum != 1 && chancesSum != 10 && chancesSum != 100){
        cout << " Warning! Chances dont add up to 100%!\n";
    }

    res = res/chancesSum;
    return res;
}

double discreteReturn(double priceNow, double priceBefore){
    return (priceNow/priceBefore) - 1;
}

double discreteReturn(double prices[MAXSIZE], double response[MAXSIZE], int size){
    int i;

    for(int i = 0; i < size-1; i++){
        response[i] = (prices[i+1]/prices[i])-1;
    }
    
    return 0;
}

double continousReturn(double priceNow, double priceBefore){
    return log(priceNow/priceBefore)/log(EulerConstant);
}

double continousReturn(double prices[MAXSIZE], double response[MAXSIZE], int size){
    int i;

    for(int i = 0; i < size-1; i++){
        response[i] = log(prices[i+1]/prices[i])/log(EulerConstant);
    }
    
    return 0;
}

double avarageDiscreteReturn(double prices[MAXSIZE], int size){
    double response[MAXSIZE], res;
    discreteReturn(prices, response, size);
    res = getAvarage(response, size-1);
    res = roundf(res * 10000) / 10000;
    return res;
}

double avarageContinousReturn(double prices[MAXSIZE], int size){
    double responseA[MAXSIZE], res;
    continousReturn(prices, responseA, size);
    res = getAvarage(responseA, size-1);
    res = roundf(res * 10000) / 10000;
    return res;
}

double avarageReturnWithPercentage(double percentage[MAXSIZE], int size){
    double res;
    res = getAvarage(percentage, size);
    res = roundf(res * 10000) / 10000;
    return res;
}

double getPossibleDiscreteReturnsWithPrices(double priceNow ,double prices[MAXSIZE], double response[MAXSIZE], int size){
    int i;
    for(i = 0; i<size; i++){
        response[i] = discreteReturn(prices[i], priceNow);
    }
    return 0;
}

double getPossibleContinousReturnsWithPrices(double priceNow ,double prices[MAXSIZE], double response[MAXSIZE], int size){
    int i;
    for(i = 0; i<size; i++){
        response[i] = continousReturn(prices[i], priceNow);
    }
    return 0;
}

double getRisk(double prices[MAXSIZE], int size){
    double response[MAXSIZE], res;
    discreteReturn(prices, response, size);
    res = getStandardDeviation(response, size-1);
    return res;
}

double getRiskWithPercentage(double percentage[MAXSIZE], int size){
    double res;
    res = getStandardDeviation(percentage, size);
    return res;
}

double getVolatility(double prices[MAXSIZE], int size){
    double response[MAXSIZE], res;
    continousReturn(prices, response, size);
    res = getStandardDeviation(response, size-1);

    return res;
}

double getRiskPerRuturn(double prices[MAXSIZE], int size){
    double response[MAXSIZE], risk, returns;
    risk = getRisk(prices, size);
    returns = avarageDiscreteReturn(prices, size);

    return risk/returns;
}

double getRiskPerRuturnWithPercentage(double percentage[MAXSIZE], int size){
    double risk, returns;
    risk = getRiskWithPercentage(percentage, size);
    returns = avarageReturnWithPercentage(percentage, size);

    return risk/returns;
}

double getVolatilityPerRuturn(double prices[MAXSIZE], int size){
    double response[MAXSIZE], risk, returns;
    risk = getVolatility(prices, size);
    returns = avarageContinousReturn(prices, size);

    return risk/returns;
}

double getAvarage(double values[MAXSIZE], int size){
    int i;
    double res = 0;

    for(int i = 0; i < size; i++){
        res = values[i] + res;
    }

    return res/size;
}

double getAvarageWithWeight(double values[MAXSIZE], double weight[MAXSIZE], int size){
    int i;
    double res = 0, weightTotal = 0;

    for(int i = 0; i < size; i++){
        res = values[i]*weight[i] + res;
        weightTotal = weightTotal + weight[i];
    }

    return res/weightTotal;
}

double getMedian(double values[MAXSIZE], int size){

    double valuesSorted[MAXSIZE];
    sortArray(values, valuesSorted, size);

    if(size%2){
        return valuesSorted[size/2];
    }

    return (valuesSorted[size/2] + valuesSorted[size/2 - 1])/2;

}

double getMode(double values[MAXSIZE], int size){
    
    double array[MAXSIZE];
    sortArray(values, array, size);
    
    double number = array[0];
    double mode = number;
    int count = 1;
    int countMode = 1;
    int i;

    for (i=1; i<size; i++)
    {
        if (array[i] == number) 
        { 
            ++count;
        }
        else
        { 
                if (count > countMode) 
                {
                    countMode = count; 
                    mode = number;
                }
            count = 1; 
            number = array[i];
    }
    }

    return mode;
    //Thanks stackoverflow
}

double getVariance(double array[MAXSIZE], int size){
    double avarage = getAvarage(array, size), res = 0;

    for(int i = 0; i < size; i++){
        res = res + pow((array[i]-avarage), 2);
    }

    return res/(size-1);
}

double getVarianceWithWeight(double array[MAXSIZE], double weight[MAXSIZE], int size){
    double avarage = getAvarageWithWeight(array, weight, size), res = 0, weightTotal = 0;

    for(int i = 0; i < size; i++){
        res = res + (pow((array[i]-avarage), 2))*weight[i];
        weightTotal = weightTotal + weight[i];
    }

    

    return res/weightTotal;
}

double getStandardDeviation(double array[MAXSIZE], int size){
    double variance = getVariance(array, size);
    return pow(variance, 0.5);
}

void sortArray(double values[MAXSIZE], double sorted[MAXSIZE], int size){
  
    for(int i = 0; i<size; i++){sorted[i] = values[i];}
    sort(sorted, sorted+size);
    
}
