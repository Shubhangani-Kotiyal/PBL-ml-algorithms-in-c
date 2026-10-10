#include <iostream>
using namespace std;

class HierarchicalClustering
{
private:
    int data[4][2];
    int group[4];

public:

    HierarchicalClustering(int d[4][2])
    {
        for (int i = 0; i < 4; i++)
        {
            data[i][0] = d[i][0];
            data[i][1] = d[i][1];
            group[i] = i;
        }
    }

    int distance(int p1, int p2)
    {
        int x = data[p1][0] - data[p2][0];
        int y = data[p1][1] - data[p2][1];
        return x * x + y * y;
    }

    void cluster()
    {
        int minDistance = 9999;
        int p1 = 0, p2 = 0;

        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                int d = distance(i, j);

                if (d < minDistance)
                {
                    minDistance = d;
                    p1 = i;
                    p2 = j;
                }
            }
        }

        for (int i = 0; i < 4; i++)
        {
            if (group[i] == group[p2])
                group[i] = group[p1];
        }

        minDistance = 9999;

        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                if (group[i] != group[j])
                {
                    int d = distance(i, j);

                    if (d < minDistance)
                    {
                        minDistance = d;
                        p1 = group[i];
                        p2 = group[j];
                    }
                }
            }
        }

        for (int i = 0; i < 4; i++)
        {
            if (group[i] == p2)
                group[i] = p1;
        }

        display();
    }

    void display()
    {
        cout << "Final Clusters:\n";

        for (int i = 0; i < 4; i++)
        {
            cout << "P" << i + 1
                 << " -> Cluster "
                 << group[i] + 1 << endl;
        }
    }
};

int main()
{
    int data[8][2] =
{
    {1, 1},   
    {2, 2},   
    {3, 3},   
    {4, 4},   
    {10, 10}, 
    {11, 11}, 
    {12, 12}, 
    {13, 13}
};

    HierarchicalClustering hc(data);

    hc.cluster();

    return 0;
}
