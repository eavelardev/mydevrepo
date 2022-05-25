#include <iostream>

using namespace std;

int main()
{
    float salePrice;
    float stateSalesTax, citySalesTax, luxuryTax;
    float salesTax, amountDue;

    cout << "Selling Price: ";
    cin >> salePrice;

    stateSalesTax = salePrice * 0.04;
    citySalesTax = salePrice * 0.015;

    if (salePrice > 50000)
        luxuryTax = salePrice * 0.1;
    else
        luxuryTax = 0;

    salesTax = stateSalesTax + citySalesTax + luxuryTax;
    amountDue = salePrice + salesTax;

    cout << "Total sales tax: " << salesTax << endl;
    cout << "Final price: " << amountDue << endl;
}
