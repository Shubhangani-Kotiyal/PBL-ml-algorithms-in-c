#include <iostream>  
#include <fstream>   
#include <cmath>     
using namespace std;

const int MAX_ROWS = 100;
const int MAX_COLS = 3;


void skipSpaces(ifstream &file) {
    while (file.peek() == ' ' || file.peek() == '\t')
        file.get();
}

int loadCsv(const char filename[], int columns, double data[][MAX_COLS]) {
    ifstream file(filename);
    if (!file) {
        cout << "Could not open " << filename << endl;
        return 0;
    }

    int n = 0;
    while (file >> ws && !file.eof()) {
        if (n == MAX_ROWS) {
            cout << "Maximum supported dataset size is 100 rows.\n";
            return 0;
        }
        for (int j = 0; j < columns; j++) {
            skipSpaces(file);
            
            if (file.peek() == '\n' || file.peek() == '\r' ||
                !(file >> data[n][j]) || !isfinite(data[n][j])) {
                cout << "Invalid CSV: use numbers only, with no header or missing values.\n";
                return 0;
            }
            skipSpaces(file);
            if (j < columns - 1) {
                char comma;
                file.get(comma);
                if (!file || comma != ',') {
                    cout << "Expected a comma between values.\n";
                    return 0;
                }
            }
            }
    
        if (!file.eof() && file.peek() != '\n' && file.peek() != '\r') {
            cout << "Extra values or characters at the end of a row.\n";
            return 0;
        }
        n++;
    }
    if (n == 0) cout << "The file contains no data rows.\n";
    return n;
}

void runLinearRegression() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("linear_regression.csv", 2, data);
    if (n == 0) return;
    if (n < 2) {
        cout << "At least two rows are required.\n";
        return;
    }

    double meanX = 0, meanY = 0;
    for (int i = 0; i < n; i++) {
        meanX += data[i][0];
        meanY += data[i][1];
    }
    meanX /= n;
    meanY /= n;

    double numerator = 0, denominator = 0;
    for (int i = 0; i < n; i++) {
        double dx = data[i][0] - meanX;
        numerator += dx * (data[i][1] - meanY);
        denominator += dx * dx;
    }
    if (denominator == 0) {
        cout << "Cannot train: all x values are the same.\n";
        return;
    }
    double slope = numerator / denominator;
    double intercept = meanY - slope * meanX;

    double totalError = 0;
    for (int i = 0; i < n; i++) {
        double prediction = slope * data[i][0] + intercept;
        double error = data[i][1] - prediction;
        totalError += error * error;
    }
    cout << "Slope: " << slope << "\nIntercept: " << intercept;
    cout << "\nMean Squared Error: " << totalError / n << endl;

    double value;
    cout << "Enter x to predict y: ";
    if (!(cin >> value) || !isfinite(value)) {
        cout << "Invalid number.\n";
        return;
    }
    cout << "Predicted y: " << slope * value + intercept << endl;
}

void runKMeans() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("kmeans.csv", 1, data);
    if (n == 0) return;

    double mean1, mean2;
    cout << "Enter the two initial centroids: ";
    if (!(cin >> mean1 >> mean2) || !isfinite(mean1) || !isfinite(mean2)) {
        cout << "Invalid centroids.\n";
        return;
    }

    for (int iteration = 0; iteration < 100; iteration++) {
        double sum1 = 0, sum2 = 0;
        int count1 = 0, count2 = 0;
        for (int i = 0; i < n; i++) {
            double value = data[i][0];
            if (fabs(value - mean1) <= fabs(value - mean2)) {
                sum1 += value;
                count1++;
            } else {
                sum2 += value;
                count2++;
            }
        }
        
        double newMean1 = mean1, newMean2 = mean2;
        if (count1 > 0) newMean1 = sum1 / count1;
        if (count2 > 0) newMean2 = sum2 / count2;
        if (newMean1 == mean1 && newMean2 == mean2) break;
        mean1 = newMean1;
        mean2 = newMean2;
    }

    cout << "Final centroid 1: " << mean1 << "\nCluster 1: ";
    for (int i = 0; i < n; i++) {
        if (fabs(data[i][0] - mean1) <= fabs(data[i][0] - mean2))
            cout << data[i][0] << ' ';
    }
    cout << "\nFinal centroid 2: " << mean2 << "\nCluster 2: ";
    for (int i = 0; i < n; i++) {
        if (fabs(data[i][0] - mean1) > fabs(data[i][0] - mean2))
            cout << data[i][0] << ' ';
    }
    cout << endl;
}
class LinearSVM {
private:
    double weights[2];
    double bias;
    double learningRate;
    double lambda;
    int iterations;

public:
    LinearSVM() {
        weights[0] = 0;
        weights[1] = 0;
        bias = 0;
        learningRate = 0.001;
        lambda = 0.01;
        iterations = 2000;
    }

    void train(double features[][2], int labels[], int n) {
        for (int iteration = 0; iteration < iterations; iteration++) {
            for (int i = 0; i < n; i++) {
                double output = weights[0] * features[i][0]
                              + weights[1] * features[i][1] + bias;
               
                if (labels[i] * output >= 1) {
                    for (int j = 0; j < 2; j++)
                        weights[j] -= learningRate * 2 * lambda * weights[j];
                } else {
                    for (int j = 0; j < 2; j++)
                        weights[j] -= learningRate *
                            (2 * lambda * weights[j] - features[i][j] * labels[i]);
                    bias += learningRate * labels[i];
                }
            }
        }
    }

    int predict(double x, double y) {
        double result = weights[0] * x + weights[1] * y + bias;
        if (result >= 0) return 1;
        return -1;
    }
};

void runSVM() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("svm.csv", 3, data);
    if (n == 0) return;

    double features[MAX_ROWS][2];
    int labels[MAX_ROWS];
    int trainingCount = 0, predictionCount = 0;
    int negativeCount = 0, positiveCount = 0;

    for (int i = 0; i < n; i++) {
        double label = data[i][2];
        if (label != -1 && label != 0 && label != 1) {
            cout << "SVM labels must be -1, 1, or 0 for prediction rows.\n";
            return;
        }
        if (label == 0) {
            predictionCount++;
        } else {
            features[trainingCount][0] = data[i][0];
            features[trainingCount][1] = data[i][1];
            labels[trainingCount] = (int)label;
            trainingCount++;
            if (label == -1) negativeCount++;
            else positiveCount++;
        }
    }
    if (negativeCount == 0 || positiveCount == 0) {
        cout << "Training data must contain both -1 and +1 classes.\n";
        return;
    }
    if (predictionCount == 0) {
        cout << "Add prediction rows with label 0 to svm.csv.\n";
        return;
    }

    LinearSVM model;
    model.train(features, labels, trainingCount);
    cout << "SVM predictions:\n";
    for (int i = 0; i < n; i++) {
        if (data[i][2] == 0)
            cout << '(' << data[i][0] << ", " << data[i][1]
                 << ") -> class " << model.predict(data[i][0], data[i][1]) << endl;
    }
}
void runHierarchicalClustering() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("hierarchical.csv", 2, data);
    if (n == 0) return;

    int target;
    cout << "Enter desired number of clusters (1 to " << n << "): ";
    if (!(cin >> target) || target < 1 || target > n) {
        cout << "Invalid cluster count.\n";
        return;
    }
    int group[MAX_ROWS];
    for (int i = 0; i < n; i++) group[i] = i;

    int activeClusters = n;
    while (activeClusters > target) {
        double bestDistance = 0;
        int bestA = -1, bestB = -1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (group[i] == group[j]) continue;
                double dx = data[i][0] - data[j][0];
                double dy = data[i][1] - data[j][1];
                double distance = dx * dx + dy * dy;
                
                if (bestA == -1 || distance < bestDistance) {
                    bestDistance = distance;
                    bestA = group[i];
                    bestB = group[j];
                }
            }
        }
        if (bestA == -1) break;
        
        for (int i = 0; i < n; i++) {
            if (group[i] == bestB) group[i] = bestA;
        }
        activeClusters--;
    }

    
    int displayId[MAX_ROWS] = {0};
    int nextId = 1;
    cout << "Clusters (single-link):\n";
    for (int i = 0; i < n; i++) {
        int g = group[i];
        if (displayId[g] == 0) {
            displayId[g] = nextId;
            nextId++;
        }
        cout << "P" << i + 1 << " (" << data[i][0] << ", " << data[i][1]
             << ") -> Cluster " << displayId[g] << endl;
    }
}
void runPCA() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("pca.csv", 2, data);
    if (n == 0) return;
    if (n < 2) {
        cout << "At least two rows are required.\n";
        return;
    }
    double meanX = 0, meanY = 0;
    for (int i = 0; i < n; i++) {
        meanX += data[i][0];
        meanY += data[i][1];
    }
    meanX /= n;
    meanY /= n;

    double covXX = 0, covXY = 0, covYY = 0;
    cout << "Mean: " << meanX << ' ' << meanY << "\nCentered data:\n";
    for (int i = 0; i < n; i++) {
        double x = data[i][0] - meanX;
        double y = data[i][1] - meanY;
        cout << x << ' ' << y << endl;
        covXX += x * x;
        covXY += x * y;
        covYY += y * y;
    }
    covXX /= (n - 1);
    covXY /= (n - 1);
    covYY /= (n - 1);
    cout << "Covariance matrix:\n";
    cout << covXX << ' ' << covXY << endl;
    cout << covXY << ' ' << covYY << endl;
    cout << "PCA preparation only: principal components are not calculated.\n";
}
void runKNN() {
    double data[MAX_ROWS][MAX_COLS];
    int n = loadCsv("knn.csv", 3, data);
    if (n == 0) return;

    double labels[MAX_ROWS];
    for (int i = 0; i < n; i++) {
        double label = data[i][2];
        if (floor(label) != label) {
            cout << "KNN labels must be whole numbers.\n";
            return;
        }
        labels[i] = label;
    }

    int k;
    cout << "Enter K (1 to " << n << "): ";
    if (!(cin >> k) || k < 1 || k > n) {
        cout << "Invalid K.\n";
        return;
    }
    double newX, newY;
    cout << "Enter the two features of the point to classify: ";
    if (!(cin >> newX >> newY) || !isfinite(newX) || !isfinite(newY)) {
        cout << "Invalid point.\n";
        return;
    }

    double distance[MAX_ROWS];
    for (int i = 0; i < n; i++) {
        double dx = newX - data[i][0];
        double dy = newY - data[i][1];
        distance[i] = sqrt(dx * dx + dy * dy);
    }

    
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (distance[j] > distance[j + 1] ||
                (distance[j] == distance[j + 1] && labels[j] > labels[j + 1])) {
                double tempDistance = distance[j];
                distance[j] = distance[j + 1];
                distance[j + 1] = tempDistance;
                double tempLabel = labels[j];
                labels[j] = labels[j + 1];
                labels[j + 1] = tempLabel;
            }
        }
    }

    double predictedClass = labels[0];
    int highestVotes = 0;
    cout << "Nearest neighbours (distance, class):\n";
    for (int i = 0; i < k; i++) {
        cout << distance[i] << ", " << labels[i] << endl;
        int votes = 0;
        for (int j = 0; j < k; j++) {
            if (labels[i] == labels[j]) votes++;
        }
        
        if (votes > highestVotes ||
            (votes == highestVotes && labels[i] < predictedClass)) {
            highestVotes = votes;
            predictedClass = labels[i];
        }
    }
    cout << "Predicted class: " << predictedClass << endl;
}
int main() {
    int choice;
    do {
        cout << "\n=== Machine Learning Algorithms ===\n";
        cout << "1. Linear Regression\n";
        cout << "2. K-means (two clusters)\n";
        cout << "3. Linear SVM\n";
        cout << "4. Hierarchical Clustering\n";
        cout << "5. PCA preparation (centering and covariance)\n";
        cout << "6. KNN\n";
        cout << "0. Exit\nChoose an algorithm: ";

        if (!(cin >> choice)) {
            if (cin.eof()) break;
            cout << "Enter a number from 0 to 6.\n";
            break;
        }
        switch (choice) {
            case 1: runLinearRegression(); break;
            case 2: runKMeans(); break;
            case 3: runSVM(); break;
            case 4: runHierarchicalClustering(); break;
            case 5: runPCA(); break;
            case 6: runKNN(); break;
            case 0: cout << "Exiting.\n"; break;
            default: cout << "Invalid choice.\n";
        }
    } while (choice != 0);
    return 0;
}
