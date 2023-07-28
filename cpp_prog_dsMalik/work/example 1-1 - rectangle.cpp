#include <iostream>

using namespace std;

int main()
{
    float length, width;
    float perimeter, area;

    cout << "Length: ";
    cin >> length;

    cout << "Width: ";
    cin >> width;

    perimeter = 2 * (length + width);
    area = length * width;

    cout << "Perimeter: " << perimeter << endl;
    cout << "Area: " << area << endl;
}
