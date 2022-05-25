#include <iostream>

using namespace std;

int main()
{
    float lenght, width;
    float perimeter, area;

    cout << "Lenght: ";
    cin >> lenght;

    cout << "Width: ";
    cin >> width;

    perimeter = 2 * (lenght + width);
    area = lenght * width;

    cout << "Perimeter: " << perimeter << endl;
    cout << "Area: " << area << endl;
}
