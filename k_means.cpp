#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>

using namespace std;

// Structure to represent a data point
struct Point {
    double x, y;
    int clusterId; // Tracks which cluster this point currently belongs to
    
    Point(double x_val, double y_val) : x(x_val), y(y_val), clusterId(-1) {}
};

// Calculate squared Euclidean distance between two points (faster than computing sqrt)
double getSquaredDistance(Point p1, Point p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

// K-Means Clustering Function
void kMeans(vector<Point>& points, int k, int maxIterations) {
    int n = points.size();
    if (n < k) return;

    // 1. Initialize Centroids randomly from existing data points
    vector<Point> centroids;
    for (int i = 0; i < k; ++i) {
        int randomIndex = rand() % n;
        centroids.push_back(points[randomIndex]);
    }

    // Iterative optimization
    for (int iter = 0; iter < maxIterations; ++iter) {
        bool changed = false;

        // 2. Assignment Step: Assign each point to the nearest centroid
        for (int i = 0; i < n; ++i) {
            double minDistance = numeric_limits<double>::max();
            int closestCluster = -1;

            for (int j = 0; j < k; ++j) {
                double dist = getSquaredDistance(points[i], centroids[j]);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestCluster = j;
                }
            }

            // Track if any point shifted clusters to check for convergence
            if (points[i].clusterId != closestCluster) {
                points[i].clusterId = closestCluster;
                changed = true;
            }
        }

        // If no points changed their cluster assignment, the algorithm has converged
        if (!changed) {
            cout << "Converged early at iteration " << iter + 1 << endl;
            break;
        }

        // 3. Update Step: Recalculate centroids based on the mean of assigned points
        vector<double> sumX(k, 0.0);
        vector<double> sumY(k, 0.0);
        vector<int> counts(k, 0);

        for (int i = 0; i < n; ++i) {
            int clusterId = points[i].clusterId;
            sumX[clusterId] += points[i].x;
            sumY[clusterId] += points[i].y;
            counts[clusterId]++;
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                centroids[j].x = sumX[j] / counts[j];
                centroids[j].y = sumY[j] / counts[j];
            }
        }
    }

    // Print final Centroids
    cout << "\nFinal Centroids:\n";
    for (int j = 0; j < k; ++j) {
        cout << "Cluster " << j << ": (" << centroids[j].x << ", " << centroids[j].y << ")\n";
    }
}

int main() {
    // Seed random number generator
    srand(time(0));

    // Sample 2D Dataset
    vector<Point> points = {
        Point(1.0, 1.0), Point(1.5, 2.0), Point(3.0, 4.0),
        Point(5.0, 7.0), Point(3.5, 5.0), Point(4.5, 5.0),
        Point(10.0, 10.0), Point(11.0, 12.0), Point(12.0, 11.0)
    };

    int k = 3;             // Number of clusters
    int maxIterations = 100; // Safety breakout limit

    cout << "Running K-Means Clustering..." << endl;
    kMeans(points, k, maxIterations);

    // Print clustered data points
    cout << "\nAssigned Points:\n";
    for (const auto& p : points) {
        cout << "Point (" << p.x << ", " << p.y << ") -> Cluster ID: " << p.clusterId << "\n";
    }

    return 0;
}
