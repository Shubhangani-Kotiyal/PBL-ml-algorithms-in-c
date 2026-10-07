#include <iostream>
using namespace std;

class LinearRegression {
private:
    double x[100], y[100];
    int n;
    double slope, intercept;

public:
    LinearRegression() {
        n = 0;
        slope = 0;
        intercept = 0;
    }

    int input() {
        cout << "Enter number of data points (2 to 100): ";
        cin >> n;

        if (!cin || n < 2 || n > 100) {
            cout << "Invalid number of data points.\n";
            return 0;
        }

        cout << "Enter x and y for each data point:\n";

        for (int i = 0; i < n; i++) {
            cout << "Point " << i + 1 << ": ";
            cin >> x[i] >> y[i];

            if (!cin) {
                cout << "Please enter numeric values.\n";
                return 0;
            }
        }

        return 1;
    }

    int train() {
        double sumX = 0, sumY = 0;

        for (int i = 0; i < n; i++) {
            sumX = sumX + x[i];
            sumY = sumY + y[i];
        }

        double meanX = sumX / n;
        double meanY = sumY / n;

        double numerator = 0;
        double denominator = 0;

        for (int i = 0; i < n; i++) {
            numerator = numerator
                      + (x[i] - meanX) * (y[i] - meanY);

            denominator = denominator
                        + (x[i] - meanX) * (x[i] - meanX);
        }

        if (denominator == 0) {
            cout << "Cannot train: all x values are the same.\n";
            return 0;
        }

        slope = numerator / denominator;
        intercept = meanY - slope * meanX;

        return 1;
    }

    double predict(double value) {
        return slope * value + intercept;
    }

    void display() {
        cout << "\nSlope: " << slope;
        cout << "\nIntercept: " << intercept;
        cout << "\nEquation: y = (" << slope
             << ") * x + (" << intercept << ")\n";
    }

    void calculateError() {
        double totalError = 0;

        for (int i = 0; i < n; i++) {
            double predicted = predict(x[i]);
            double error = y[i] - predicted;

            totalError = totalError + error * error;
        }

        cout << "Mean Squared Error: " << totalError / n << endl;
    }
};

int main() {
    LinearRegression model;

    if (model.input() == 0) {
        return 1;
    }

    if (model.train() == 0) {
        return 1;
    }

    model.display();
    model.calculateError();

    double value;

    cout << "\nEnter x to predict y: ";
    cin >> value;

    if (!cin) {
        cout << "Please enter a numeric value.\n";
        return 1;
    }

    cout << "Predicted y: " << model.predict(value) << endl;

    return 0;
}
