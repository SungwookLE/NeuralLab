#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "readCSV.h"

using namespace std;

class RNNs
{
public:
    RNNs(int _nStates, int _nUnits)
    {
        nStates = _nStates;
        nUnits = _nUnits;
        init();
    }

    void init()
    {
        // loop: hidden state
        H = vector<double>(nUnits, 0);

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0, 0.05);

        Wx = vector<vector<double>>(nStates, vector<double>(nUnits, 0));
        Wh = vector<vector<double>>(nUnits, vector<double>(nUnits, 0));

        Bx = vector<double>(nUnits, 0);
        Wy = vector<double>(nUnits, 0);

        By = dist(gen);
        for (int i = 0; i < nStates; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
            {
                Wx[i][j] = dist(gen);
            }
        }

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
            {
                Wh[i][j] = dist(gen);
            }

            Bx[i] = dist(gen);
            Wy[i] = dist(gen);
        }
    }

    void backward(vector<double> X, double Y)
    {
        // save the gradients

        /*
         net_Y = H*Wy + By
         hat_Y = tanh(net_Y)
         net_H = X*Wx + prev_H*Wh + Bx
         H = tanh(net_H)

         Loss = 0.5*(hat_Y-Y)^2
         1)
         aL      aL        a(hat_Y)   a(net_Y)
         ----  = ------- x -------- x -------- = (hat_Y-Y) * 1 * H
         a(Wy)   a(hat_Y)  a(net_Y)   a(Wy)

         aL      aL        a(hat_Y)   a(net_Y)
         ----  = ------- x -------- x -------- = (hat_Y-Y) * 1 * 1
         a(By)   a(hat_Y)  a(net_Y)   a(By)

         2)BPTT
         aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
         ----  = -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1  * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * X + ....
         a(Wx)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Wx)

         aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
         ----  = -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1  * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * 1 + ....
         a(Bx)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Bx)

         3)BPTT
         aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
         ---- =  -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1  * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * prev_H + ....
         a(Wh)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Wh)
        */
        vector<double> prev_H = H;
        double net_H[nUnits] = {
            0,
        };
        double out_H[nUnits] = {
            0,
        };

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nStates; ++j)
                net_H[i] += X[j] * Wx[j][i];
        }

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
                net_H[i] += prev_H[i] * Wh[i][j];
        }

        for (int i = 0; i < nUnits; ++i)
            net_H[i] += Bx[i];

        for (int i = 0; i < nUnits; ++i)
            out_H[i] = tanh(net_H[i]);

        double net_Y = 0;
        for (int i = 0; i < nUnits; ++i)
        {
            net_Y += out_H[i] * Wy[i];
        }
        net_Y = net_Y + By;

        double hat_Y = net_Y;

        /*
        1)
          aL      aL        a(hat_Y)   a(net_Y)
          ----  = ------- x -------- x -------- = (hat_Y-Y) * 1 *H
          a(Wy)   a(hat_Y)  a(net_Y)   a(Wy)

          aL      aL        a(hat_Y)   a(net_Y)
          ----  = ------- x -------- x -------- = (hat_Y-Y) * 1 *1
          a(By)   a(hat_Y)  a(net_Y)   a(By)
        */
        double aL_aWy[nUnits] = {
            0,
        },
               aL_aBy = 0;

        for (int i = 0; i < nUnits; ++i)
        {
            aL_aWy[i] = (hat_Y - Y) * 1 * out_H[i];
            Wy[i] = Wy[i] - lr * aL_aWy[i];
        }
        aL_aBy = (hat_Y - Y) * 1 * 1;
        By = By - lr * aL_aBy;

        /*
        2)BPTT
          aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
          ----  = -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1 * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * X + ....
          a(Wx)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Wx)

          aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
          ----  = -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1 * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * 1 + ....
          a(Bx)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Bx)
        */

        double aL_aWx[nStates][nUnits] = {
            0,
        },
               aL_aBx[nUnits] = {
                   0,
               };

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nStates; ++j)
            {
                aL_aWx[j][i] = (hat_Y - Y) * 1 * Wy[i] * (1 - tanh(net_H[i])) * (1 + tanh(net_H[i])) * X[j];
                Wx[j][i] = Wx[j][i] - lr * aL_aWx[j][i];
            }
            aL_aBx[i] = (hat_Y - Y) * 1 * Wy[i] * (1 - tanh(net_H[i])) * (1 + tanh(net_H[i])) * 1;
            Bx[i] = Bx[i] - lr * aL_aBx[i];
        }

        /*
         3)BPTT
           aL      aL         a(hat_Y)   a(net_Y)   aH         a(net_H)
           ---- =  -------- x -------- x -------- x -------- x -------- + ...... = (hat_Y-Y) * 1 * Wy * (1-tanh(net_H)) * (1+tanh(net_H)) * prev_H + ....
           a(Wh)   a(hat_Y)   a(net_Y)   aH         a(net_H)   a(Wh)
        */
        double aL_aWh[nUnits][nUnits] = {
            0,
        };
        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
            {
                aL_aWh[i][j] = (hat_Y - Y) * 1 * Wy[i] * (1 - tanh(net_H[i])) * (1 + tanh(net_H[i])) * prev_H[i];
                Wh[i][j] = Wh[i][j] - lr * aL_aWh[i][j];
            }
        }

        return;
    }

    double forward_step(vector<double> X, vector<double> &H)
    {

        double hat_Y, temp_Y = 0;
        vector<double> temp_H(nUnits, 0);
        vector<double> temp_X(nUnits, 0);

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
            {
                temp_H[i] += Wh[i][j] * H[i];
            }

            for (int j = 0; j < nStates; ++j)
                temp_X[i] += X[j] * Wx[j][i];

            H[i] = tanh(temp_X[i] + Bx[i] + temp_H[i]);
            temp_Y += H[i] * Wy[i];
        }

        hat_Y = temp_Y + By;
        return hat_Y;
    }

    vector<double> inference(vector<vector<double>> X)
    {

        vector<double> res;
        this->H = vector<double>(nUnits, 0);
        for (auto x : X)
        {
            res.push_back(forward_step(x, this->H));
            cout << res.back() << ", ";
        }

        return res;
    }

    void train(vector<vector<double>> X, vector<double> Y, int iter)
    {

        for (int i = 0; i < iter; ++i)
        {
            this->H = vector<double>(nUnits, 0);
            for (int j = 0; j < X.size(); ++j)
            {
                this->forward_step(X[j], this->H);
                this->backward(X[j], Y[j]);
            }
        }

        return;
    }

    vector<double> H;

private:
    int nStates, nUnits;
    vector<vector<double>> Wx, Wh;
    vector<double> Bx, Wy;
    double By;
    double lr = 0.002;
};

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

int main()
{
    vector<double> time, acc, pos, vel_gt;
    read_csv("data/dataSet.csv", time, acc, pos, vel_gt);

    RNNs model(2, 20);

    vector<vector<double>> X;
    X.push_back(time);
    X.push_back(acc);
    X.push_back(pos);
    transpose(X);

    model.train(X, vel_gt, 100);

    vector<double> hat_Y;
    hat_Y = model.inference(X);

    save_csv("result/rnn.csv", hat_Y, vel_gt);

    return 0;
}