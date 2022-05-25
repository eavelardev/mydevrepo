#include <iostream>

using namespace std;

int main()
{
    int num, guess;

    cout << "Number: ";
    cin >> num;

    do
    {
        cout << "Enter guess number: ";
        cin >> guess;

        if (guess == num)
            cout << "You guessed the correct number." << endl;
        else
            if (guess < num)
                cout << "Your guess is lower than the number. Guess again!" << endl;
            else
                cout << "Your guess is higher than the number. Guess again!" << endl;
                
    } while (guess != num);
    
}