#include <iostream>
using namespace std;

class PCA
{
private:
    int data[4][2];
    double mean[2];
    double centered[4][2];
    double covariance[2][2];

public:

    PCA(int d[4][2])
    {
        for (int i = 0; i < 4; i++)
        {
            data[i][0] = d[i][0];
            data[i][1] = d[i][1];
        }
    }

    void calculateMean()
    {
        mean[0] = 0;
        mean[1] = 0;

        for (int i = 0; i < 4; i++)
        {
            mean[0] += data[i][0];
            mean[1] += data[i][1];
        }

        mean[0] = mean[0] / 4;
        mean[1] = mean[1] / 4;
    }

    void centerData()
    {
        for (int i = 0; i < 4; i++)
        {
            centered[i][0] = data[i][0] - mean[0];
            centered[i][1] = data[i][1] - mean[1];
        }
    }

    void calculateCovariance()
    {
        covariance[0][0] = 0;
        covariance[0][1] = 0;
        covariance[1][0] = 0;
        covariance[1][1] = 0;

        for (int i = 0; i < 4; i++)
        {
            covariance[0][0] +=
                centered[i][0] * centered[i][0];

            covariance[0][1] +=
                centered[i][0] * centered[i][1];

            covariance[1][0] +=
                centered[i][1] * centered[i][0];

            covariance[1][1] +=
                centered[i][1] * centered[i][1];
        }

        covariance[0][0] = covariance[0][0] / 3;
        covariance[0][1] = covariance[0][1] / 3;
        covariance[1][0] = covariance[1][0] / 3;
        covariance[1][1] = covariance[1][1] / 3;
    }

    void display()
    {
        cout << "Mean:\n";
        cout << mean[0] << " " << mean[1] << "\n\n";

        cout << "Centered Data:\n";

        for (int i = 0; i < 4; i++)
        {
            cout << centered[i][0] << " "
                 << centered[i][1] << endl;
        }

        cout << "\nCovariance Matrix:\n";

        cout << covariance[0][0] << " "
             << covariance[0][1] << endl;

        cout << covariance[1][0] << " "
             << covariance[1][1] << endl;
    }

    void run()
    {
        calculateMean();
        centerData();
        calculateCovariance();
        display();
    }
};

int main()
{
    int data[8][2] =
{
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
    {5, 10},
    {6, 12},
    {7, 14},
    {8, 16}
};

    PCA pca(data);

    pca.run();

    return 0;
}
