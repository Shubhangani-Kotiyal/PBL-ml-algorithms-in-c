#include <iostream>
#include <cmath>
using namespace std;

class KNN {
private:
    double x[100], y[100];
    int label[100];
    int n, k;

public:
    KNN() {
        n = 0;
        k = 0;
    }

    int input() {
        cout << "Enter number of training points (1 to 100): ";
        cin >> n;

        if (!cin || n < 1 || n > 100) {
            cout << "Invalid number of points.\n";
            return 0;
        }

        cout << "Enter two features and class label for each point:\n";

        for (int i = 0; i < n; i++) {
            cout << "Point " << i + 1 << ": ";
            cin >> x[i] >> y[i] >> label[i];

            if (!cin) {
                cout << "Invalid input.\n";
                return 0;
            }
        }

        cout << "Enter value of K: ";
        cin >> k;

        if (!cin || k < 1 || k > n) {
            cout << "K must be between 1 and " << n << ".\n";
            return 0;
        }

        return 1;
    }

    int predict(double newX, double newY) {
        double distance[100];
        int neighbourLabel[100];

        // Step 1: Calculate distance from every training point.
        for (int i = 0; i < n; i++) {
            double dx = newX - x[i];
            double dy = newY - y[i];

            distance[i] = sqrt(dx * dx + dy * dy);
            neighbourLabel[i] = label[i];
        }

        // Step 2: Sort distances using bubble sort.
        // Move each label along with its distance.
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (distance[j] > distance[j + 1]) {
                    double tempDistance = distance[j];
                    distance[j] = distance[j + 1];
                    distance[j + 1] = tempDistance;

                    int tempLabel = neighbourLabel[j];
                    neighbourLabel[j] = neighbourLabel[j + 1];
                    neighbourLabel[j + 1] = tempLabel;
                }
            }
        }

        cout << "\nK nearest neighbours:\n";
        cout << "Distance\tClass\n";

        for (int i = 0; i < k; i++) {
            cout << distance[i] << "\t\t"
                 << neighbourLabel[i] << endl;
        }

        // Step 3: Count votes for each class among the first K points.
        int predictedClass = neighbourLabel[0];
        int highestVotes = 0;

        for (int i = 0; i < k; i++) {
            int votes = 0;

            for (int j = 0; j < k; j++) {
                if (neighbourLabel[i] == neighbourLabel[j]) {
                    votes++;
                }
            }

            if (votes > highestVotes) {
                highestVotes = votes;
                predictedClass = neighbourLabel[i];
            }
        }

        return predictedClass;
    }
};

int main() {
    KNN model;

    if (model.input() == 0) {
        return 1;
    }

    double newX, newY;

    cout << "\nEnter two features of the new point: ";
    cin >> newX >> newY;

    if (!cin) {
        cout << "Please enter numeric values.\n";
        return 1;
    }

    int result = model.predict(newX, newY);

    cout << "\nPredicted class: " << result << endl;

    return 0;
}
