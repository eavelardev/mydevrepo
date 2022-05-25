#include <iostream>

using namespace std;

int main()
{
    const int num_tests = 5;
    int scores[num_tests], average, sum = 0;
    char grade;

    // class average not implemented
    //int num_students = 10;

    // Get the score
    for (int i = 0; i < num_tests; i++)
    {
        cout << "Test score " << i+1 << ": ";
        cin >> scores[i];
        sum += scores[i];
    }

    average = (int)(sum / num_tests);

    if (average >= 90)
        grade = 'A';
    else
        if (average >= 80)
            grade = 'B';
        else
            if (average >= 70)
                grade = 'C';
            else
                if (average >= 60)
                    grade = 'D';
                else
                    grade = 'F';
    
    cout << "Average: " << average << endl;
    cout << "Grade: " << grade << endl;
}
