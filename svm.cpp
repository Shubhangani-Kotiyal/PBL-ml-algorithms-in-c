#include <iostream>
#include <vector>

// Using the standard namespace to avoid typing "std::" repeatedly
using namespace std;

class LinearSVM {
private:
    vector<double> weights;
    double bias;
    double learning_rate;
    double lambda_param; 
    int iterations;

public:
    LinearSVM(double lr = 0.001, double lambda = 0.01, int iters = 1000)
        : bias(0.0), learning_rate(lr), lambda_param(lambda), iterations(iters) {}

    void train(const vector<vector<double>>& X, const vector<int>& y) {
        int num_samples = X.size();
        int num_features = X[0].size(); // Fixed: features should map to the inner vector size
        
        weights.assign(num_features, 0.0);
        bias = 0.0;

        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < num_samples; ++i) {
                double linear_output = bias;
                for (int j = 0; j < num_features; ++j) {
                    linear_output += weights[j] * X[i][j];
                }

                if (y[i] * linear_output >= 1) {
                    for (int j = 0; j < num_features; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j]);
                    }
                } else {
                    for (int j = 0; j < num_features; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j] - X[i][j] * y[i]);
                    }
                    bias += learning_rate * y[i];
                }
            }
        }
    }

    int predict(const vector<double>& x) const {
        double approximation = bias;
        for (size_t i = 0; i < x.size(); ++i) {
            approximation += weights[i] * x[i];
        }
        return (approximation >= 0) ? 1 : -1;
    }
};

int main() {
    vector<vector<double>> X = {
        {1.0, 2.0}, {2.0, 1.0}, {2.0, 3.0}, 
        {6.0, 5.0}, {7.0, 8.0}, {8.0, 6.0}  
    };
    vector<int> y = {-1, -1, -1, 1, 1, 1};

    LinearSVM svm(0.001, 0.01, 1000);
    svm.train(X, y);

    vector<vector<double>> test_points = {
        {1.5, 1.5}, 
        {7.5, 7.5}  
    };

    cout << "--- SVM Predictions ---" << endl;
    for (const auto& point : test_points) {
        int prediction = svm.predict(point);
        // Corrected print logic to access vector elements directly
        cout << "Point (" << point[0] << ", " << point[1] 
             << ") -> Predicted Class: " << prediction << endl;
    }

    return 0;
}
