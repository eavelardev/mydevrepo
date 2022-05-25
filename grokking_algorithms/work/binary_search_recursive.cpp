#include <iostream>

using namespace std;

bool binary_search(int data_array[], int low, int high, int elem)
{
    if (low <= high)
    {
        int mid = (low + high) / 2;
        int guess = data_array[mid];

        if (guess == elem)
            return true;
        
        if (guess < elem)
            return binary_search(data_array, mid + 1, high, elem);
        else
            return binary_search(data_array, low, mid - 1, elem);
    }

    return false;
}

int main()
{
    int data_array[] = {2,10,23,44,100,121}, elem;
    int lenght = sizeof(data_array) / sizeof(data_array[0]);
    int low = 0;
    int high = lenght - 1;

    elem = 10;

    if (binary_search(data_array, low, high, elem))
        cout << elem << " found" << endl;
    else
        cout << elem << " not found" << endl;
}
