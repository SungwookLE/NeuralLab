#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include <cmath>
#include <random>

void read_csv(std::string file_names, std::vector<std::vector<double>> &group_a, std::vector<std::vector<double>> &group_b){
    std::ifstream filestream(file_names);
    if(filestream.is_open()){
        int idx =0;
        std::string line;
        while(std::getline(filestream, line)){
            std::replace(line.begin(), line.end(), ',',' ');
            std::istringstream linestream(line);
            std::string x_data, y_data, label;
            linestream >> x_data >> y_data >> label;
            std::vector<double> res;
            res = {std::stod(x_data), std::stod(y_data), std::stod(label)};
            if (idx <100)
                group_a.push_back(res);
            else
                group_b.push_back(res);
            idx+=1;
        }
    }
    return;
}

double activate_sigmoid(double in){
    double res;
    res = (double)1.0/(1.0+std::exp(-in));

    return res;
}

double activate_relu(double in){
    double res;

    if (in>=0)
        res = in;
    else
        res = 0.0;

    return res;
}

double activate_step(double in){
    double res;

    if (in>=0)
        res = 1.0;
    else
        res = 0.0;

    return res;
}

double forward_propagation(const std::vector<double> X, const std::vector<double> W, const double B, const std::string opt){
    double res;
    double ans;

    res = X.at(0)*W[0] + X.at(1)*W[1] + B;

    if (opt == "relu"){
        ans = activate_relu(res);
    }
    else if (opt == "step"){
        ans = activate_step(res);
    }
    else // (opt == "sigmoid")
    {
        ans = activate_sigmoid(res);
    }    

    return ans;
}

void backward_propagation(const double output, const double label, const std::vector<double> X, std::vector<double> &W, double &B, double learning_rate, const std::string opt){
    
    double gradient_W0, gradient_W1, gradient_B;

    if (opt=="sigmoid"){
        gradient_W0 = -(label - output)*(1 - output)*output * X[0];
        gradient_W1 = -(label - output)*(1 - output)*output * X[1];
        gradient_B =  -(label - output)*(1 - output)*output * 1;

        W[0] = W[0] - learning_rate * gradient_W0;
        W[1] = W[1] - learning_rate * gradient_W1;
        B = B - learning_rate * gradient_B;
    }

    return;
}

int main(){

    /* 0-1. Random Number Gen */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0,0.25);
    
    /* 0-2. Data Set Ready:
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
    read_csv("dataset.csv",group_a, group_b);
    std::vector<std::vector<double>> data_set;
    data_set.insert(data_set.end(), group_a.begin(), group_a.end());
    data_set.insert(data_set.end(), group_b.begin(), group_b.end());
    
    /* 1. Initialization of Node weights and bias */
    std::vector<double> W = {dist(gen), dist(gen)}; //weights
    double B = dist(gen); //bias
    
    /* 2. Learning Step with Backpropagation */
    int iteration= 50;
    double learning_rate =0.01;
    for(int iter=0; iter < iteration; ++iter){
        for(int i=0; i<data_set.size(); ++i){
            std::vector<double> res = {data_set.at(i)[0], data_set.at(i)[1]};
            //2-1. FORWARD PROPAGATION
            double out=forward_propagation(res, W, B, "sigmoid");
            //2-2. BACKWARD PROPAGATION
            backward_propagation(out, data_set.at(i)[2], res, W, B, learning_rate, "sigmoid");
        }   
    }
    
    std::cout << "W[0]:" << W[0] << ", "<< "W[1]:" << W[1] << ", " << "B:" << B << std::endl;
    /* 3. Validation with Forwardpropagation */
    std::vector<double> out_net;
    for(int i=0; i<data_set.size(); ++i){
        std::vector<double> res = {data_set.at(i)[0], data_set.at(i)[1]};
        //3-1. FORWARD PROPAGATION
        double out=forward_propagation(res, W, B, "sigmoid");
        out_net.push_back(out);
    }
    
    /* 4. SCORE with CSV export */
    int score =0;
    for(int i =0 ; i < data_set.size(); ++i){
        int label_est = std::round(out_net[i]);
        if (label_est == data_set[i][2])
            score+=1;
    }

    std::cout << "VALIDATION(" << score << "/"<< out_net.size() << "): " << (double)score/out_net.size()*100 <<"%"<<std::endl;
    return 0;
}

// https://github.com/SungwookLE/udacity_TensorFlow_LAB/blob/master/README_backpropa.MD