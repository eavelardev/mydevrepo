#include <iostream>

using namespace std;

bool binary_search(int data_array[], int high, int elem)
{
    int low = 0;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        int guess = data_array[mid];

        if (guess == elem)
            return true;
        
        if (guess < elem)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return false;
}

int main()
{
    int data_array[] = {2,10,23,44,100,121}, elem;
    int lenght = sizeof(data_array) / sizeof(data_array[0]);
    int high = lenght - 1;

    elem = 10;

    if (binary_search(data_array, high, elem))
        cout << elem << " found" << endl;
    else
        cout << elem << " not found" << endl;
}
