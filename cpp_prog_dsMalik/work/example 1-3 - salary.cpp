#include <iostream>

using namespace std;

int main()
{
    int totalSales;
    float baseSalary, noOfServiceYears, bonus, additionalBonus, payCheck;

    cout << "Base Salary: ";
    cin >> baseSalary;

    cout << "Number of years: ";
    cin >> noOfServiceYears;

    cout << "Total Sales: ";
    cin >> totalSales;

    if (noOfServiceYears <= 5)
        bonus = 10 * noOfServiceYears;
    else
        bonus = 20 * noOfServiceYears;

    if (totalSales < 5000)
        additionalBonus = 0;
    else
        if (totalSales >= 5000 && totalSales < 10000)
            additionalBonus = totalSales * 0.03;
        else
            additionalBonus = totalSales * 0.06;

    payCheck = baseSalary + bonus + additionalBonus;

    cout << "Pay Check: " << payCheck;
}
