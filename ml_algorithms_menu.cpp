#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// CSV utilities: rows must contain numeric fields separated by commas.
// A single non-numeric first row is accepted as a header.
string trim(const string& value) {
    const string whitespace = " \t\r\n";
    size_t first = value.find_first_not_of(whitespace);
    if (first == string::npos) return "";
    size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

bool parseNumber(const string& text, double& value) {
    string cleaned = trim(text);
    if (cleaned.empty()) return false;
    try {
        size_t used = 0;
        value = stod(cleaned, &used);
        return used == cleaned.size();
    } catch (...) {
        return false;
    }
}

bool loadCsv(const string& filename, size_t expectedColumns,
             vector<vector<double>>& rows) {
    ifstream file(filename);
    if (!file) {
        cerr << "Could not open " << filename << ".\n";
        return false;
    }

    rows.clear();
    string line;
    int lineNumber = 0;
    bool headerConsidered = false;

    while (getline(file, line)) {
        ++lineNumber;
        if (trim(line).empty()) continue;

        vector<double> values;
        string cell;
        stringstream stream(line);
        bool valid = true;

        while (getline(stream, cell, ',')) {
            double number;
            if (!parseNumber(cell, number)) {
                valid = false;
                break;
            }
            values.push_back(number);
        }

        if (valid && values.size() == expectedColumns) {
            rows.push_back(values);
            headerConsidered = true;
        } else if (!headerConsidered && rows.empty()) {
            // Permit one header row, for example: x,y or feature1,feature2,label.
            headerConsidered = true;
        } else {
            cerr << "Invalid CSV row " << lineNumber << " in " << filename
                 << ". Expected " << expectedColumns << " numeric column(s).\n";
            return false;
        }
    }

    if (rows.empty()) {
        cerr << "No data rows found in " << filename << ".\n";
        return false;
    }
    return true;
}

void runLinearRegression() {
    vector<vector<double>> rows;
    if (!loadCsv("linear_regression.csv", 2, rows) || rows.size() < 2) {
        if (rows.size() < 2) cerr << "Linear regression needs at least two rows.\n";
        return;
    }

    double meanX = 0.0, meanY = 0.0;
    for (const auto& row : rows) {
        meanX += row[0];
        meanY += row[1];
    }
    meanX /= rows.size();
    meanY /= rows.size();

    double numerator = 0.0, denominator = 0.0;
    for (const auto& row : rows) {
        numerator += (row[0] - meanX) * (row[1] - meanY);
        denominator += (row[0] - meanX) * (row[0] - meanX);
    }
    if (denominator == 0.0) {
        cerr << "Cannot train: all x values are the same.\n";
        return;
    }

    double slope = numerator / denominator;
    double intercept = meanY - slope * meanX;
    double mse = 0.0;
    for (const auto& row : rows) {
        double error = row[1] - (slope * row[0] + intercept);
        mse += error * error;
    }

    cout << "Loaded " << rows.size() << " rows from linear_regression.csv\n";
    cout << "Slope: " << slope << "\nIntercept: " << intercept
         << "\nMean Squared Error: " << mse / rows.size() << '\n';

    double value;
    cout << "Enter x to predict y: ";
    if (cin >> value) cout << "Predicted y: " << slope * value + intercept << '\n';
    else { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); cerr << "Invalid number.\n"; }
}

void runKMeans() {
    vector<vector<double>> rows;
    if (!loadCsv("kmeans.csv", 1, rows)) return;

    double mean1, mean2;
    cout << "Enter initial centroid 1: ";
    if (!(cin >> mean1)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); cerr << "Invalid centroid.\n"; return; }
    cout << "Enter initial centroid 2: ";
    if (!(cin >> mean2)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); cerr << "Invalid centroid.\n"; return; }

    vector<double> cluster1, cluster2;
    for (int iter = 0; iter < 100; ++iter) {
        cluster1.clear();
        cluster2.clear();
        double sum1 = 0.0, sum2 = 0.0;

        for (const auto& row : rows) {
            double value = row[0];
            if (fabs(value - mean1) <= fabs(value - mean2)) {
                cluster1.push_back(value); sum1 += value;
            } else {
                cluster2.push_back(value); sum2 += value;
            }
        }

        double newMean1 = cluster1.empty() ? mean1 : sum1 / cluster1.size();
        double newMean2 = cluster2.empty() ? mean2 : sum2 / cluster2.size();
        if (newMean1 == mean1 && newMean2 == mean2) break;
        mean1 = newMean1;
        mean2 = newMean2;
    }

    // Reassign after the final centroid update so output matches the final means.
    cluster1.clear(); cluster2.clear();
    for (const auto& row : rows) {
        if (fabs(row[0] - mean1) <= fabs(row[0] - mean2)) cluster1.push_back(row[0]);
        else cluster2.push_back(row[0]);
    }

    cout << "Final centroid 1: " << mean1 << "\nCluster 1: ";
    for (double value : cluster1) cout << value << ' ';
    cout << "\nFinal centroid 2: " << mean2 << "\nCluster 2: ";
    for (double value : cluster2) cout << value << ' ';
    cout << '\n';
}

class LinearSVM {
private:
    double weights[2] = {0.0, 0.0};
    double bias = 0.0;
    double learningRate;
    double lambda;
    int iterations;

public:
    LinearSVM(double lr = 0.001, double regularization = 0.01, int steps = 2000)
        : learningRate(lr), lambda(regularization), iterations(steps) {}

    void train(const vector<vector<double>>& features, const vector<int>& labels) {
        for (int iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < features.size(); ++i) {
                double output = bias + weights[0] * features[i][0] + weights[1] * features[i][1];
                if (labels[i] * output >= 1) {
                    for (int j = 0; j < 2; ++j)
                        weights[j] -= learningRate * 2 * lambda * weights[j];
                } else {
                    for (int j = 0; j < 2; ++j)
                        weights[j] -= learningRate * (2 * lambda * weights[j] - features[i][j] * labels[i]);
                    bias += learningRate * labels[i];
                }
            }
        }
    }

    int predict(const vector<double>& point) const {
        return bias + weights[0] * point[0] + weights[1] * point[1] >= 0 ? 1 : -1;
    }
};

void runSVM() {
    vector<vector<double>> rows;
    if (!loadCsv("svm.csv", 3, rows)) return;

    vector<vector<double>> trainingFeatures, predictionPoints;
    vector<int> trainingLabels;
    bool hasNegative = false, hasPositive = false;

    for (const auto& row : rows) {
        int label = static_cast<int>(row[2]);
        if (row[2] != label || (label != -1 && label != 0 && label != 1)) {
            cerr << "SVM labels must be -1, 1, or 0 (prediction row).\n";
            return;
        }
        if (label == 0) predictionPoints.push_back({row[0], row[1]});
        else {
            trainingFeatures.push_back({row[0], row[1]});
            trainingLabels.push_back(label);
            if (label == -1) hasNegative = true;
            if (label == 1) hasPositive = true;
        }
    }

    if (!hasNegative || !hasPositive) {
        cerr << "SVM training rows must include both -1 and 1 labels.\n";
        return;
    }
    if (predictionPoints.empty()) {
        cerr << "Add prediction rows with label 0 to svm.csv.\n";
        return;
    }

    LinearSVM model;
    model.train(trainingFeatures, trainingLabels);
    cout << "SVM predictions:\n";
    for (const auto& point : predictionPoints)
        cout << '(' << point[0] << ", " << point[1] << ") -> class " << model.predict(point) << '\n';
}

void runHierarchicalClustering() {
    vector<vector<double>> rows;
    if (!loadCsv("hierarchical.csv", 2, rows)) return;

    int targetClusters;
    cout << "Enter the desired number of clusters (1 to " << rows.size() << "): ";
    if (!(cin >> targetClusters) || targetClusters < 1 || targetClusters > static_cast<int>(rows.size())) {
        cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cerr << "Invalid cluster count.\n";
        return;
    }

    vector<int> group(rows.size());
    for (size_t i = 0; i < group.size(); ++i) group[i] = static_cast<int>(i);

    // Single-link agglomerative clustering using squared Euclidean distance.
    int activeClusters = static_cast<int>(rows.size());
    while (activeClusters > targetClusters) {
        double bestDistance = numeric_limits<double>::infinity();
        int bestA = -1, bestB = -1;
        for (size_t i = 0; i < rows.size(); ++i) {
            for (size_t j = i + 1; j < rows.size(); ++j) {
                if (group[i] == group[j]) continue;
                double dx = rows[i][0] - rows[j][0];
                double dy = rows[i][1] - rows[j][1];
                double distance = dx * dx + dy * dy;
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestA = group[i];
                    bestB = group[j];
                }
            }
        }
        if (bestA < 0 || bestB < 0) break;
        for (int& cluster : group) if (cluster == bestB) cluster = bestA;
        --activeClusters;
    }

    map<int, int> displayId;
    int nextId = 1;
    cout << "Clusters (single-link):\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (displayId.find(group[i]) == displayId.end()) displayId[group[i]] = nextId++;
        cout << "P" << i + 1 << " (" << rows[i][0] << ", " << rows[i][1]
             << ") -> Cluster " << displayId[group[i]] << '\n';
    }
}

void runPCA() {
    vector<vector<double>> rows;
    if (!loadCsv("pca.csv", 2, rows) || rows.size() < 2) {
        if (rows.size() < 2) cerr << "PCA covariance requires at least two rows.\n";
        return;
    }

    double mean1 = 0.0, mean2 = 0.0;
    for (const auto& row : rows) { mean1 += row[0]; mean2 += row[1]; }
    mean1 /= rows.size();
    mean2 /= rows.size();

    double cov00 = 0.0, cov01 = 0.0, cov11 = 0.0;
    cout << "Mean: " << mean1 << ' ' << mean2 << "\n\nCentered data:\n";
    for (const auto& row : rows) {
        double x = row[0] - mean1;
        double y = row[1] - mean2;
        cout << x << ' ' << y << '\n';
        cov00 += x * x;
        cov01 += x * y;
        cov11 += y * y;
    }
    double divisor = static_cast<double>(rows.size() - 1);
    cov00 /= divisor; cov01 /= divisor; cov11 /= divisor;

    cout << "\nCovariance matrix:\n" << cov00 << ' ' << cov01 << '\n'
         << cov01 << ' ' << cov11 << '\n';
}

void runKNN() {
    vector<vector<double>> rows;
    if (!loadCsv("knn.csv", 3, rows)) return;

    vector<vector<double>> features;
    vector<int> labels;
    for (const auto& row : rows) {
        int label = static_cast<int>(row[2]);
        if (row[2] != label) { cerr << "KNN labels must be integers.\n"; return; }
        features.push_back({row[0], row[1]});
        labels.push_back(label);
    }

    int k;
    cout << "Loaded " << rows.size() << " training rows. Enter K (1 to " << rows.size() << "): ";
    if (!(cin >> k) || k < 1 || k > static_cast<int>(rows.size())) {
        cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cerr << "Invalid K.\n";
        return;
    }

    double newX, newY;
    cout << "Enter the two features of the point to classify: ";
    if (!(cin >> newX >> newY)) {
        cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cerr << "Invalid point.\n";
        return;
    }

    vector<pair<double, int>> neighbours;
    for (size_t i = 0; i < features.size(); ++i) {
        double dx = newX - features[i][0];
        double dy = newY - features[i][1];
        neighbours.push_back({sqrt(dx * dx + dy * dy), labels[i]});
    }
    sort(neighbours.begin(), neighbours.end());

    map<int, int> votes;
    cout << "Nearest neighbours (distance, class):\n";
    for (int i = 0; i < k; ++i) {
        cout << neighbours[i].first << ", " << neighbours[i].second << '\n';
        ++votes[neighbours[i].second];
    }

    int predictedClass = votes.begin()->first;
    int highestVotes = -1;
    for (const auto& vote : votes) {
        if (vote.second > highestVotes) {
            highestVotes = vote.second;
            predictedClass = vote.first;
        }
    }
    cout << "Predicted class: " << predictedClass << '\n';
}

int main() {
    while (true) {
        cout << "\n=== Machine Learning Algorithms ===\n"
             << "1. Linear Regression\n"
             << "2. K-means\n"
             << "3. Linear SVM\n"
             << "4. Hierarchical Clustering\n"
             << "5. PCA\n"
             << "6. KNN\n"
             << "0. Exit\n"
             << "Choose an algorithm: ";

        int choice;
        if (!(cin >> choice)) {
            if (cin.eof()) break;
            cin.clear();
            cin.ignore(numeric_limits<streamsize>::max(), '\n');
            cerr << "Enter a number from 0 to 6.\n";
            continue;
        }

        switch (choice) {
            case 1: runLinearRegression(); break;
            case 2: runKMeans(); break;
            case 3: runSVM(); break;
            case 4: runHierarchicalClustering(); break;
            case 5: runPCA(); break;
            case 6: runKNN(); break;
            case 0: cout << "Exiting.\n"; return 0;
            default: cout << "Invalid choice. Select 0 to 6.\n";
        }
    }
    return 0;
}
