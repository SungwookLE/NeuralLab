#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "readCSV.h"

using namespace std;

namespace preprocessing
{
    void normalized(vector<double> &X, double scaled)
    {

        double sumX = 0;
        for (int i = 0; i < X.size(); ++i)
            sumX += X[i] * X[i];

        double meanSumX = sumX / X.size();
        double rootMeanSumX = sqrt(meanSumX);

        for (int i = 0; i < X.size(); ++i)
            X[i] = X[i] / rootMeanSumX * scaled;

        return;
    }

    void transpose(vector<vector<double>> &X)
    {
        vector<vector<double>> trans_X(X[0].size(), vector<double>());
        for (int i = 0; i < X.size(); ++i)
        {
            for (int j = 0; j < X[i].size(); ++j)
            {
                trans_X[j].push_back(X[i][j]);
            }
        }
        X = trans_X;
        return;
    }

    void transpose_many(vector<vector<double>> &X, int manyWindow = 1)
    {
        vector<vector<double>> trans_X(X[0].size(), vector<double>());
        for (int i = 0; i < X.size(); ++i)
        {
            for (int j = 0; j < X[i].size(); ++j)
            {
                if ((j + 1) >= manyWindow)
                    for (int k = 0; k < manyWindow; ++k)
                    {
                        trans_X[j].push_back(X[i][j - k]);
                    }
                else
                {
                    for (int k = j; k >= 0; --k)
                    {
                        trans_X[j].push_back(X[i][k]);
                    }
                    for (int k = 0; k < (manyWindow - j - 1); ++k)
                    {
                        trans_X[j].push_back(0);
                    }
                }
            }
        }
        X = trans_X;
        return;
    }
}
