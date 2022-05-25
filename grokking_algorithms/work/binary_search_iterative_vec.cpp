#include <iostream>
#include <vector>

using namespace std;

bool binary_search(vector<int> data, int elem)
{
    int low = 0;
    int high = data.size() - 1;

    while (low <= high)
    {
        int mid = (low + high) / 2;
        int guess = data[mid];

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
    vector<int> data = {2,10,23,44,100,121};
    int elem = 23;

    if (binary_search(data, elem))
        cout << elem << " found" << endl;
    else
        cout << elem << " not found" << endl;
}
