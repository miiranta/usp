#include <iostream>
#include <iomanip>
#include "handler.hpp"
#define MAXSIZE 100
using namespace std;

int main(){

    //SET VALUES!

        //ONE ASSET

            //PAST INFO 

                //Price variation Stats
                double prices[]  = {25.6,25.3,27.51,30.43}; //t1 t2 t3 t4...

                //Percentage variation Stats
                double percentage[] = {-5,6,-4,8,10,4,3.17}; //Write in PERCENTAGE, not decimal

            //FUTURE INFO (((MORE TESTING REQUIRED)))

                //Price expected return Stats 
                double priceNow = 1;
                double prices2[] = { 0.8,    1.05,    1.3}; 
                double chances[] = { 0.1,  0.3,  0.6}; //Write in DECIMAL (Chances for each price)

                //Percentage expected return Stats 
                double percentage2[] = { -20,    5,   30}; //Write in PERCENTAGE, not decimal
                double chances2[]    = { 0.1,  0.3,  0.6}; //Write in DECIMAL

        //TEST
        double numbers[] = {1, 2, 2, 3, 11, 7, 4, 9};
        double weight[]  = {1, 4, 1, 8,  1, 8, 8, 8};
 

    //Ignore
    double response[MAXSIZE], res;
    int sizePrices = sizeof(prices)/sizeof(prices[0]);
    int sizePrices2 = sizeof(prices2)/sizeof(prices2[0]);
    int sizeNumbers = sizeof(numbers)/sizeof(numbers[0]);
    int sizePercentage = sizeof(percentage)/sizeof(percentage[0]);
    int sizePercentage2 = sizeof(percentage2)/sizeof(percentage2[0]);
    cout.precision(4);

    //UNCOMMENT!
        //oneAssetPastByPrices(prices, sizePrices);
        //oneAssetPastByPercentages(percentage, sizePercentage);

        //oneAssetFutureByPrices(priceNow, prices2, chances, sizePrices2);
        //oneAssetFutureByPercentages(percentage2, chances2, sizePercentage2);


 









}