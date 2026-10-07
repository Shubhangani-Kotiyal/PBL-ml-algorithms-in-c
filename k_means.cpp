#include <iostream>

using namespace std;

int main() {
    int data[10];
    int cluster1[10], cluster2[10];
    int count1, count2;
    int mean1, mean2;

    // Input 10 numbers
    cout << "Enter 10 numbers:\n";
    for (int i = 0; i < 10; i++) {
        cin >> data[i];
    }

    // Input initial means (centroids)
    cout << "Enter initial mean 1: ";
    cin >> mean1;
    cout << "Enter initial mean 2: ";
    cin >> mean2;

    // Run K-Means for a fixed number of iterations
    for (int iter = 0; iter < 100; iter++) {
        count1 = 0;
        count2 = 0;
        int sum1 = 0, sum2 = 0;

        // Assign points to the nearest mean
        for (int i = 0; i < 10; i++) {
            int dist1 = (data[i] > mean1) ? (data[i] - mean1) : (mean1 - data[i]);
            int dist2 = (data[i] > mean2) ? (data[i] - mean2) : (mean2 - data[i]);

            if (dist1 <= dist2) {
                cluster1[count1++] = data[i];
                sum1 += data[i];
            } else {
                cluster2[count2++] = data[i];
                sum2 += data[i];
            }
        }

        // Recalculate means
        int newMean1 = (count1 > 0) ? (sum1 / count1) : mean1;
        int newMean2 = (count2 > 0) ? (sum2 / count2) : mean2;

        // Check for convergence
        if (newMean1 == mean1 && newMean2 == mean2) {
            break;
        }

        mean1 = newMean1;
        mean2 = newMean2;
    }

    // Output final results
    cout << "\nFinal Mean 1: " << mean1 << "\nCluster 1 elements: ";
    for (int i = 0; i < count1; i++) {
        cout << cluster1[i] << " ";
    }

    cout << "\n\nFinal Mean 2: " << mean2 << "\nCluster 2 elements: ";
    for (int i = 0; i < count2; i++) {
        cout << cluster2[i] << " ";
    }
    cout << endl;

    return 0;
}
