#include <iostream>
#include <iomanip>
#include "functions.hpp"
#define MAXSIZE 100
using namespace std;

double res, ris, ret, response[MAXSIZE];
int i;

//PAST
void oneAssetPastByPrices(double prices[MAXSIZE], int sizePrices){
    
    cout << "\nONE ASSET - PAST - BY PRICES ====================\n\n";
    
        cout << "                         ";
        for(i = 0; i<sizePrices;i++){cout <<"t"<< i <<"      ";}
        cout << "\nPrices over time:   ";
        printArray(prices, sizePrices, 0);
        
        cout << "\n                         ";
        for(i = 0; i<sizePrices;i++){cout <<"t"<< i <<" ---> ";}
        cout << "\nDiscrete Returns:        ";
        discreteReturn(prices, response, sizePrices);
        printArray(response, sizePrices-1, 1);

        cout << "Continous Returns:       ";
        continousReturn(prices, response, sizePrices);
        printArray(response, sizePrices-1, 1);

        res = avarageDiscreteReturn(prices, sizePrices);
        cout << "\nAvarage discrete return: " << res*100 << "%\n";

        res = getRisk(prices, sizePrices);
        cout << "Risk: " << res*100 << "%\n";

        res = getRiskPerRuturn(prices, sizePrices);
        cout << "Risk/Return: " << res << "\n";

        res = avarageContinousReturn(prices, sizePrices);
        cout << "\nAvarage continous return: " << res*100 << "%\n";

        res = getVolatility(prices, sizePrices);
        cout << "Volatility: " << res*100 << "%\n";

        res = getVolatilityPerRuturn(prices, sizePrices);
        cout << "Volatility/Return: " << res << "\n";

    cout << "\n=================================================\n\n";

}

void oneAssetPastByPercentages(double percentages[MAXSIZE], int sizePercentages){

    cout << "\nONE ASSET - PAST - BY PERCENTAGES ===============\n\n";
    
        cout << "                ";
        for(i = 0; i<sizePercentages+1;i++){cout <<"t"<< i <<" ---> ";}
        cout << "\nReturns:        ";
        printArray(percentages, sizePercentages, 2);

        res = avarageReturnWithPercentage(percentages, sizePercentages);
        cout << "\nAvarage discrete return: " << res << "%\n";

        res = getRiskWithPercentage(percentages, sizePercentages);
        cout << "Risk: " << res << "%\n";

        res = getRiskPerRuturnWithPercentage(percentages, sizePercentages);
        cout << "Risk/Return: " << res << "\n";


    cout << "\n=================================================\n\n";

}

//FUTURE

void oneAssetFutureByPrices(double priceNow, double prices[MAXSIZE], double chances[MAXSIZE], int sizePrices){

    cout << "\nONE ASSET - FUTURE - BY PRICES ==================\n\n";

        cout << setw(7) << "Price Now:  " << priceNow << "\n";

        cout << "Possible prices in the future: ";
        printArray(prices, sizePrices, 0);

        cout << "\nPossible discrete returns:     ";
        getPossibleDiscreteReturnsWithPrices(priceNow , prices, response, sizePrices);
        printArray(response, sizePrices, 1);

        cout << "Chances:                       ";
        printArray(chances, sizePrices, 1);

        ret = expectedReturn(response, chances, sizePrices);
        cout << setw(7)  << "Expected discrete return:  " << ret*100 << "%\n";

        ris = expectedRisk(response, chances, sizePrices);
        cout << "Expected discrete risk:    " << ris*100 << "%\n"; 

        cout << "Risk/Return:               " << (ris/ret)  << "\n";
        
        cout << "\nPossible continous returns:    ";
        getPossibleContinousReturnsWithPrices(priceNow , prices, response, sizePrices);
        printArray(response, sizePrices, 1);

        cout << "Chances:                       ";
        printArray(chances, sizePrices, 1);

        ret = expectedReturn(response, chances, sizePrices);
        cout << setw(7)  << "Expected continous return: " << ret*100 << "%\n";

        ris = expectedRisk(response, chances, sizePrices);
        cout << "Expected continous risk:   " << ris*100 << "%\n";  
        
        cout << "Risk/Return:               " << (ris/ret) << "\n";

    cout << "\n=================================================\n\n";

}

void oneAssetFutureByPercentages(double percentages[MAXSIZE], double chances[MAXSIZE], int sizePercentages){

    cout << "\nONE ASSET - FUTURE - BY PERCENTAGES =============\n\n";

        cout << "Returns: ";
        printArray(percentages, sizePercentages, 2);

        cout << "Chances: ";
        printArray(chances, sizePercentages, 1);

        ret = expectedReturn(percentages, chances, sizePercentages);
        cout << "\nExpected return: " << ret << "%\n";

        ris = expectedRisk(percentages, chances, sizePercentages);
        cout << "Expected risk:   " << ris << "%\n";

        cout << "Risk/Return:     " << (ris/ret) << "\n";

    cout << "\n=================================================\n\n";

}

//TEST
void getNumberStats(double avarage[MAXSIZE], double weight[MAXSIZE], int sizeAvarage){

    cout << "\n================================\n\n";

        cout << "Numbers:  ";
        printArray(avarage, sizeAvarage, 0);

        cout << "Weight:   ";
        printArray(weight, sizeAvarage, 0); 

        sortArray(avarage, response, sizeAvarage);
        cout << "\nSorted:   ";
        printArray(response, sizeAvarage, 0); 

        res = getAvarage(avarage, sizeAvarage);
        cout << "\nAvarage: " << res;

        res = getAvarageWithWeight(avarage, weight, sizeAvarage);
        cout << "\nAvarage with weight: " << res;

        res = getMedian(avarage, sizeAvarage);
        cout << "\nMedian: " << res;

        res = getMode(avarage, sizeAvarage);
        cout << "\nMode (first): " << res;

        res = getVariance(avarage, sizeAvarage);
        cout << "\nVariance: " << res;

        res = getStandardDeviation(avarage, sizeAvarage);
        cout << "\nStandard deviation: " << res;

    cout << "\n\n================================\n\n";

}

