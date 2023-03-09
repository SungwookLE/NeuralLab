// g++ build Command(terminal): g++ singleLayer.cpp readCSV.cpp -o single.exe

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include "readCSV.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// TO-DO 1: return sigmoid value
double activate_sigmoid(double in){
    
    double res
    return res;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

double activate_relu(double in){
    double res;

    if (in>=0)
        res = in;
    else
        res = 0.0;

    return res;
}

double forward_inferencing(const std::vector<double> X, const std::vector<double> W, const double B, const std::string opt){
    double res;
    double ans;

    res = X[0]*W[0] + X[1]*W[1] + B;

    if (opt == "relu"){
        ans = activate_relu(res);
    }
    else if (opt == "sigmoid"){
        ans = activate_sigmoid(res);
    }
   
    return ans;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// TO-DO 2: Fill the gradient descent equation
void backward_propagation(const double output, const double label, const std::vector<double> X, std::vector<double> &W, double &B, double learning_rate, const std::string opt){
    
    double gradient_W0, gradient_W1, gradient_B;

    if (opt=="sigmoid"){
        gradient_W0 = 
        gradient_W1 = 
        gradient_B =  

    }
    else if (opt == "relu"){
            gradient_W0 = 
            gradient_W1 = 
            gradient_B = 

    }

    return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
    
    unordered_map<string, double> initWeight;
    unordered_map<string, double> finalWeight;

    /* 1. Data Set Read:
        DATA STRUCTURE
        {   {X1, X2, LABEL}
            {X1, X2, LABEL}
            {X1, X2, LABEL}
            ...
        }
        GROUP_A HAS LABEL '1', GROUP HAS LABEL '0'
    */
    std::vector<std::vector<double>> group_a;
    std::vector<std::vector<double>> group_b;
    read_csv("./data/dataSet.csv", group_a, group_b);
    std::vector<std::vector<double>> data_set;
    data_set.insert(data_set.end(), group_a.begin(), group_a.end());
    data_set.insert(data_set.end(), group_b.begin(), group_b.end());
    
    /* 2. Random Number Gen */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0,0.25);

    /* 2-1. Initialization of Node weights and bias */
    std::vector<double> W = {dist(gen), dist(gen)}; //weights
    double B = dist(gen); //bias
    std::cout << "Initial={\"W[0]\":" << W[0] << ", "<< "\"W[1]\":" << W[1] << ", " << "\"B\":" << B << "}"<< std::endl;
    initWeight["W[0]"] = W[0];
    initWeight["W[1]"] = W[1];
    initWeight["B"] = B;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TO-DO 3: fill the Leaning logic 
    /* 3. Learning with Backpropagation */
    int iteration= 500;
    double learning_rate =0.05;

    for(int iter=0; iter < iteration; ++iter){
        for(int i=0; i<data_set.size(); ++i){
            // batch size: 1
            std::vector<double> feed_data = {data_set[i][0], data_set[i][1]};
            double feed_label = data_set[i][2];

            // 3-1. FORWARD Inferencing
            
            // 3-2. BACKWARD PROPAGATION
            
        }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MONITOR
        if (iter%100==0){
            /* DEBUG 1: Forward Inferencing */
            std::vector<double> out;
            for (int i = 0; i < data_set.size(); ++i) {
                std::vector<double> feed_data = { data_set[i][0], data_set[i][1] };
                double result = forward_inferencing(feed_data, W, B, "sigmoid");
                out.push_back(result);
            }
            /* DEBUG 2: Check SCORE */
            int score = 0;
            for (int i = 0; i < data_set.size(); ++i) {
                int label_est = std::round(out[i]);
                if (label_est == data_set[i][2])
                    score += 1;
            }
            /*DEBUG 3: PRINT */
            std::cout << iter<< "th ITERATION(" << score << "/" << out.size() << "): " << (double)score / out.size() * 100 << "%" << std::endl;
        }   
    }

    std::cout << "Result={\"W[0]\":" << W[0] << ", "<< "\"W[1]\":" << W[1] << ", " << "\"B\":" << B << "}"<< std::endl;
    finalWeight["W[0]"] = W[0];
    finalWeight["W[1]"] = W[1];
    finalWeight["B"] = B;
    std::cout << "DONE!!!" << std::endl;

    /* 4. Forward Inferencing */
    std::vector<double> out;
    for(int i=0; i<data_set.size(); ++i){
        std::vector<double> feed_data = {data_set[i][0], data_set[i][1]};
        //3-1. FORWARD PROPAGATION
        double result=forward_inferencing(feed_data, W, B, "sigmoid");
        out.push_back(result);
    }
    
    /* 5. SCORE with CSV export */
    int score =0; 
    for(int i =0 ; i < data_set.size(); ++i){
        int label_est = std::round(out[i]);
        if (label_est == data_set[i][2])
            score+=1;
    }
    std::vector<double> label_a, label_b;
    label_a.insert(label_a.begin(), out.begin(), out.begin()+group_a.size());
    label_b.insert(label_b.begin(), out.begin()+group_a.size(), out.end());

    save_csv("results/resultSingle.csv", group_a, group_b, label_a, label_b, initWeight, finalWeight);
    std::cout << "VALIDATION(" << score << "/"<< out.size() << "): " << (double)score/out.size()*100 <<"%"<<std::endl;

    return 0;
}