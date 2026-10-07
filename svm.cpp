#include <iostream>

using namespace std;

// Define global constants for our specific dataset size
const int NUM_SAMPLES = 6;
const int NUM_FEATURES = 2;

class LinearSVM {
private:
    double weights[NUM_FEATURES];
    double bias;
    double learning_rate;
    double lambda_param;
    int iterations;

public:
    LinearSVM(double lr = 0.001, double lambda = 0.01, int iters = 1000)
        : bias(0.0), learning_rate(lr), lambda_param(lambda), iterations(iters) {
        // Initialize weights to zero
        for (int i = 0; i < NUM_FEATURES; ++i) {
            weights[i] = 0.0;
        }
    }

    // Train the model using raw 2D arrays
    void train(const double X[NUM_SAMPLES][NUM_FEATURES], const int y[NUM_SAMPLES]) {
        bias = 0.0;

        for (int iter = 0; iter < iterations; ++iter) {
            for (int i = 0; i < NUM_SAMPLES; ++i) {
                double linear_output = bias;
                for (int j = 0; j < NUM_FEATURES; ++j) {
                    linear_output += weights[j] * X[i][j];
                }

                // Hinge Loss Gradient Step
                if (y[i] * linear_output >= 1) {
                    for (int j = 0; j < NUM_FEATURES; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j]);
                    }
                } else {
                    for (int j = 0; j < NUM_FEATURES; ++j) {
                        weights[j] -= learning_rate * (2 * lambda_param * weights[j] - X[i][j] * y[i]);
                    }
                    bias += learning_rate * y[i];
                }
            }
        }
    }

    // Predict the class label for a single 1D array sample
    int predict(const double x[NUM_FEATURES]) const {
        double approximation = bias;
        for (int i = 0; i < NUM_FEATURES; ++i) {
            approximation += weights[i] * x[i];
        }
        return (approximation >= 0) ? 1 : -1;
    }
};

int main() {
    // Dataset initialized using raw multidimensional arrays
    double X[NUM_SAMPLES][NUM_FEATURES] = {
        {1.0, 2.0}, {2.0, 1.0}, {2.0, 3.0}, // Class -1
        {6.0, 5.0}, {7.0, 8.0}, {8.0, 6.0}  // Class 1
    };
    int y[NUM_SAMPLES] = {-1, -1, -1, 1, 1, 1};

    // Initialize and train SVM
    LinearSVM svm(0.001, 0.01, 2000);
    svm.train(X, y);

    // Test items
    double test_point_1[NUM_FEATURES] = {1.5, 1.5};
    double test_point_2[NUM_FEATURES] = {7.5, 7.5};

    cout << "--- SVM Predictions ---" << endl;
    
    cout << "Point (" << test_point_1[0] << ", " << test_point_1[1] 
         << ") -> Predicted Class: " << svm.predict(test_point_1) << endl;
         
    cout << "Point (" << test_point_2[0] << ", " << test_point_2[1] 
         << ") -> Predicted Class: " << svm.predict(test_point_2) << endl;

    return 0;
}
