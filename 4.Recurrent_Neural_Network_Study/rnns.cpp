#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "readCSV.h"

using namespace std;

class RNNs
{
public:
    RNNs(int _nStates, int _nUnits, double _lr)
    {
        nStates = _nStates;
        nUnits = _nUnits;
        lr = _lr;
        init();
        cleanMem();
    }

    void cleanMem(){
        // loop: hidden state
        H = vector<vector<double>>(memorySize, vector<double>(nUnits, 0)); // 10 step memory
        Xmem = vector<vector<double>>(memorySize, vector<double>(nStates, 0)); // 10 step memory
    }

    void init()
    {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0, 0.1);

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

    void backward(double hat_Y, double Y)
    {
        /////////////////////////////////////////////
        // 1. Wy, By update
        /////////////////////////////////////////////
        double aLt_aWy[nUnits] = {0,};
        double aLt_aBy = 0;

        for(int i = 0 ; i < nUnits; ++i)
            aLt_aWy[i] = (hat_Y-Y) * H[0][i];
        aLt_aBy = (hat_Y-Y) * 1;

        for(int i = 0 ; i < nUnits; ++i)
            Wy[i] = Wy[i] - lr * aLt_aWy[i];
        By = By - lr * aLt_aBy;
        /////////////////////////////////////////////

        /////////////////////////////////////////////
        // 2. Wx, Bx update (Timewindow: 3)
        /////////////////////////////////////////////
        double aLt_aWx[nStates][nUnits] = {0,};
        double aLt_aBx[nUnits] = {0,};

        for(int i = 0 ; i < nUnits; ++i){
            double sum_Wh =0;
            for(int k = 0 ; k < nUnits; ++k)
                sum_Wh += Wh[i][k];

            for(int j =0 ; j < nStates; ++j){
                aLt_aWx[j][i] = (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * Xmem[0][j] ;
                                // + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i]) * (1+H[1][i]) * Xmem[1][j]
                                // + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i]) * (1+H[1][i]) * sum_Wh * (1-H[2][i])*(1+H[2][i]) * Xmem[2][j];

                Wx[j][i] = Wx[j][i] - lr * aLt_aWx[j][i];
            }
            aLt_aBx[i] = (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * 1;
                        // + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i]) * (1+H[1][i]) * 1
                        // + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i]) * (1+H[1][i]) * sum_Wh * (1-H[2][i])*(1+H[2][i]) * 1; 

            Bx[i] = Bx[i] - lr * aLt_aBx[i];                
        }
        /////////////////////////////////////////////

        /////////////////////////////////////////////
        // 3. Wh update (Timewindow: 3)
        /////////////////////////////////////////////
        double aLt_aWh[nUnits][nUnits] = {0,};

        for(int i = 0 ; i < nUnits; ++i){
            double sum_Wh =0;
            for(int k =0 ; k < nUnits; ++k)
                sum_Wh += Wh[i][k];


            for(int j =0 ; j < nUnits; ++j){
                aLt_aWh[i][j] = (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * H[1][j] ;
                            //   + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i])*(1+H[1][i]) * H[2][j]
                            //   + (hat_Y-Y) * Wy[i] * (1-H[0][i])*(1+H[0][i]) * sum_Wh * (1-H[1][i])*(1+H[1][i]) * sum_Wh * (1-H[2][i])*(1+H[2][i]) * H[3][j];

                Wh[i][j] = Wh[i][j] - lr * aLt_aWh[i][j];
            }
        }
        /////////////////////////////////////////////

    }

    double forward_step(vector<double> X)
    {
        double hat_Y, temp_Y = 0;
        vector<double> temp_H(nUnits, 0);
        vector<double> temp_X(nUnits, 0);
        vector<double> next_H(nUnits, 0);

        for(int p = memorySize-1 ; p > 0; --p)
            Xmem[p] = Xmem[p-1];
        Xmem[0] = X;

        for (int i = 0; i < nUnits; ++i)
        {
            for (int j = 0; j < nUnits; ++j)
                temp_H[i] += Wh[i][j] * H[0][j];

            for (int j = 0; j < nStates; ++j)
                temp_X[i] += X[j] * Wx[j][i];

            next_H[i] = tanh(temp_X[i] + Bx[i] + temp_H[i]);
            temp_Y += next_H[i] * Wy[i];
            
        }
        for(int p = memorySize-1 ; p > 0; --p)
            H[p] = H[p-1];
        H[0] = next_H;
        

        hat_Y = temp_Y + By;
        return hat_Y;
    }

    vector<double> inference(vector<vector<double>> X)
    {
        vector<double> res;
        cleanMem();
        for (auto x : X){
            res.push_back(forward_step(x));
        }
        return res;
    }

    void train(vector<vector<double>> X, vector<double> Y, int iter)
    {
        for (int i = 0; i < iter; ++i)
        {
            cleanMem();
            for (int j = 0; j < X.size(); ++j)
            {
                double hat_Y = forward_step(X[j]); 
                this->backward(hat_Y, Y[j]);
            }
        }

        return;
    }

    void monitoring(vector<double> result, vector<double> Y){
        cout << "H: " << endl;
        for(auto r : H[0]){
            cout <<  r << ", ";
        }
        cout << endl;


        cout << "HAT_Y: " << endl;
        for(auto r : result){
            cout <<  r << ", ";
        }
        cout << endl;

        double sumE  =0 ;
        for(int i =0 ; i < Y.size(); ++i){
            sumE+=(result[i] - Y[i])*(result[i] - Y[i]);
        }
        cout << "meanSquareError: " << sumE/Y.size() << endl;

        

    }

    

private:
    int nStates, nUnits;
    vector<vector<double>> Wx, Wh;
    vector<double> Bx, Wy;
    double By;
    double lr;
    vector<vector<double>> H;
    vector<vector<double>> Xmem;
    int memorySize = 10;

};

int main()
{
    vector<double> time, acc, pos, vel_gt;
    read_csv("data/dataSet.csv", time, acc, pos, vel_gt);

    RNNs model(2, 200, 0.0002);

    vector<vector<double>> X;
    X.push_back(acc);
    X.push_back(pos); // 아 acc 와 스케일링 정도를 맞추니까 된다? (4/5)
    transpose(X);

    model.train(X, vel_gt, 20);

    vector<double> hat_Y;
    hat_Y = model.inference(X);
    model.monitoring(hat_Y, vel_gt);

    save_csv("result/rnn.csv", hat_Y, vel_gt);

    return 0;
}